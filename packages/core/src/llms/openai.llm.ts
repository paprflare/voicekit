import OpenAI, {
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    RateLimitError,
    APIError,
    APIConnectionError,
} from "openai";
import { URL } from "url";
import { configureLogger } from "../helper/logger";
import { DEFAULT_LANGUAGE_CODE, GPT5_MODEL_PREFIX } from "../constants";
import { nowMs } from "../helper/utils";
import { OpenAICompatibleLLM } from "./openaiBase";
import { ToolCallAccumulator } from "./toolcall.accumulator";
import type { LLMStreamChunk, LatencyData } from "./types";
import { ReasoningEffort, Verbosity } from "../enums";

const logger = configureLogger("openAiLlm");

const JSON_MODE_MODELS = new Set([
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-1106",
    "gpt-4o-mini",
    "gpt-4.1-mini",
]);

interface OpenAiLLMOptions {
    api_tools?: {
        tools_params: Record<string, Record<string, unknown>>;
        tools: Record<string, unknown>[];
    } | null;
    language?: string;
    reasoning_effort?: string;
    verbosity?: string;
    service_tier?: string;
    provider?: string;
    base_url?: string;
    llm_key?: string;
    assistant_id?: string;
    run_id?: string;
    use_responses_api?: boolean;
}

export class OpenAiLLM extends OpenAICompatibleLLM {
    protected asyncClient: OpenAI;
    protected override model: string;
    protected override temperature: number;
    protected override language: string;
    protected override runId: string;
    protected override modelArgs: Record<string, unknown>;
    protected override triggerFunctionCall: boolean;
    protected override tools: Record<string, unknown>[] | string;
    protected override apiParams: Record<string, Record<string, unknown>>;
    protected override startedStreaming = false;
    protected override llmHost: string | undefined;

    private openaiSync?: OpenAI;
    private assistantId?: string;

    constructor(
        model = "gpt-3.5-turbo-16k",
        maxTokens = 100,
        bufferSize = 40,
        temperature = 0.1,
        language = DEFAULT_LANGUAGE_CODE,
        kwargs: OpenAiLLMOptions = {}
    ) {
        super(maxTokens, bufferSize);
        this.model = model;
        this.temperature = temperature;
        this.language = language;
        this.runId = kwargs.run_id ?? "";

        // ------------------------------------------------------------------
        // Tools
        // ------------------------------------------------------------------
        const customTools = kwargs.api_tools ?? null;
        logger.info(`API Tools ${JSON.stringify(customTools)}`);
        if (customTools) {
            this.triggerFunctionCall = true;
            this.apiParams = customTools.tools_params;
            this.tools = customTools.tools;
            logger.info(`Function dict ${JSON.stringify(this.apiParams)}`);
        } else {
            this.triggerFunctionCall = false;
            this.apiParams = {};
            this.tools = [];
        }

        // ------------------------------------------------------------------
        // Model args
        // ------------------------------------------------------------------
        let maxTokensKey = "max_tokens";
        this.modelArgs = {};

        if (model.startsWith(GPT5_MODEL_PREFIX)) {
            maxTokensKey = "max_completion_tokens";
            this.modelArgs["reasoning_effort"] =
                kwargs.reasoning_effort ?? ReasoningEffort.low;
            this.modelArgs["verbosity"] = kwargs.verbosity ?? Verbosity.low;
        }

        this.modelArgs = {
            ...this.modelArgs,
            [maxTokensKey]: maxTokens,
            temperature,
            model,
            service_tier: kwargs.service_tier ?? "default",
        };

        logger.info(
            `Initializing OpenAI LLM with model: ${this.model} and max tokens ${maxTokens}`
        );

        // ------------------------------------------------------------------
        // HTTP client (openai SDK manages its own connection pooling in Node)
        // ------------------------------------------------------------------
        const provider = kwargs.provider ?? "openai";
        const baseUrl = kwargs.base_url;
        let apiKey: string | undefined;

        if (provider === "custom") {
            apiKey = kwargs.llm_key ?? undefined;
            this.asyncClient = new OpenAI({
                baseURL: baseUrl,
                apiKey,
                maxRetries: 0,
                timeout: 600_000,
            });
        } else {
            apiKey = kwargs.llm_key ?? process.env.OPENAI_API_KEY;
            this.asyncClient = new OpenAI({
                ...(baseUrl ? { baseURL: baseUrl } : {}),
                apiKey,
                maxRetries: 0,
                timeout: 600_000,
            });
        }

        this.llmHost = baseUrl ? new URL(baseUrl).host : undefined;

        // ------------------------------------------------------------------
        // Assistant (optional)
        // ------------------------------------------------------------------
        this.assistantId = kwargs.assistant_id;
        if (this.assistantId) {
            logger.info(
                `Initializing OpenAI assistant with assistant id ${this.assistantId}`
            );
            this.openaiSync = new OpenAI({ apiKey });
            this.modelArgs = {
                max_completion_tokens: maxTokens,
                temperature,
            };
            // Note: assistant tool retrieval is async; call initAssistant() after construction if needed
        }

        this.initResponsesApi(kwargs.use_responses_api ?? false);
    }

    /** Call after construction if assistantId is set. */
    async initAssistant(): Promise<void> {
        if (!this.assistantId || !this.openaiSync) return;
        const assistant = await this.openaiSync.beta.assistants.retrieve(
            this.assistantId
        );
        if (assistant.tools) {
            this.tools = assistant.tools
                .filter((t) => t.type === "function")
                .map((t) => t as unknown as Record<string, unknown>);
        }
    }

    // ------------------------------------------------------------------
    // Public generate API
    // ------------------------------------------------------------------

    async *generateStream(
        messages: Record<string, unknown>[],
        synthesize = true,
        requestJson = false,
        metaInfo: Record<string, unknown> | undefined = undefined,
        toolChoice?: string
    ): AsyncGenerator<LLMStreamChunk> {
        if (this.useResponsesApi) {
            yield* this.generateStreamResponses(
                messages,
                synthesize,
                requestJson,
                metaInfo,
                toolChoice
            );
        } else {
            yield* this._generateStreamChat(
                messages,
                synthesize,
                requestJson,
                metaInfo,
                toolChoice
            );
        }
    }

    override async generate(
        messages: Record<string, unknown>[],
        stream = false,
        retMetadata = false
    ): Promise<unknown> {
        if (this.useResponsesApi) {
            return this.generateResponses(messages, false, retMetadata);
        }
        return this._generateChat(messages, false, retMetadata);
    }

    // ------------------------------------------------------------------
    // Chat completions — streaming
    // ------------------------------------------------------------------

    private async *_generateStreamChat(
        messages: Record<string, unknown>[],
        synthesize = true,
        requestJson = false,
        metaInfo: Record<string, unknown> | null = null,
        toolChoice?: string
    ): AsyncGenerator<LLMStreamChunk> {
        if (!messages.length) throw new Error("No messages provided");

        const responseFormat = this.getResponseFormat(requestJson);
        const modelArgs: Record<string, unknown> = {
            ...this.modelArgs,
            response_format: responseFormat,
            messages,
            stream: true,
            user: `${this.runId}#${metaInfo?.["turn_id"]}`,
        };

        if (!this.model.startsWith(GPT5_MODEL_PREFIX)) {
            modelArgs["stop"] = ["User:"];
        }

        const parsedTools =
            typeof this.tools === "string"
                ? (JSON.parse(this.tools) as Record<string, unknown>[])
                : this.tools;

        if (this.triggerFunctionCall) {
            modelArgs["tools"] = parsedTools;
            modelArgs["tool_choice"] = toolChoice ?? "auto";
            modelArgs["parallel_tool_calls"] = false;
        }

        let answer = "";
        let buffer = "";
        const accumulator = this.triggerFunctionCall
            ? new ToolCallAccumulator({
                apiParams: this.apiParams,
                tools: parsedTools,
                language: this.language,
                model: this.model,
                runId: this.runId,
            })
            : null;

        const startTime = nowMs();
        let firstTokenTime: number | null = null;
        let latencyData: LatencyData | null = null;
        let serviceTier: string | null = null;

        let completionStream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>;

        const streamParams = {
            ...modelArgs,
            stream: true as const,
        } as OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming;

        try {
            completionStream = await this.asyncClient.chat.completions.create(streamParams);
        } catch (e) {
            this._handleOpenAIError(e);
            throw e;
        }

        for await (const chunk of completionStream) {
            const now = nowMs();

            if ("service_tier" in chunk && chunk.service_tier) {
                serviceTier = chunk.service_tier as string;
                if (latencyData) latencyData.service_tier = serviceTier;
            }

            if (!firstTokenTime) {
                firstTokenTime = now;
                this.startedStreaming = true;
                latencyData = {
                    sequence_id: metaInfo
                        ? (metaInfo["sequence_id"] as number)
                        : null,
                    first_token_latency_ms: firstTokenTime - startTime,
                    total_stream_duration_ms: null,
                    service_tier: serviceTier,
                    llm_host: this.llmHost ?? null,
                };
            }

            const delta = chunk.choices[0]?.delta;
            if (!delta) continue;

            if (delta.tool_calls?.length && accumulator) {
                if (buffer) {
                    yield { data: buffer, end_of_stream: true, latency: latencyData, is_function_call: false };
                    buffer = "";
                }

                accumulator.processDelta(
                    delta.tool_calls as unknown as Array<{
                        index: number;
                        id?: string;
                        function: { name?: string; arguments?: string };
                    }>
                );

                const preCall = accumulator.getPreCallMessage(metaInfo);
                if (preCall) {
                    yield {
                        data: preCall[0],
                        end_of_stream: true,
                        latency: latencyData,
                        is_function_call: false,
                        function_name: preCall[1],
                        function_message: preCall[2] as string | Record<string, unknown>,
                    };
                }
            } else if (delta.content != null) {
                if (accumulator) accumulator.receivedTextual = true;
                answer += delta.content;
                buffer += delta.content;

                if (synthesize && buffer.length >= this.bufferSize) {
                    const parts = buffer.split(/ (?=[^ ]*$)/);
                    yield { data: parts[0], end_of_stream: false, latency: latencyData, is_function_call: false };
                    buffer = parts[1] ?? "";
                }
            }
        }

        if (latencyData) {
            latencyData.total_stream_duration_ms = nowMs() - startTime;
        }

        if (accumulator && Object.keys(accumulator["finalToolCalls"]).length) {
            const apiCallPayload = accumulator.buildApiPayload(
                modelArgs,
                metaInfo ?? {},
                answer
            );
            if (apiCallPayload) {
                yield { data: apiCallPayload, end_of_stream: false, latency: latencyData, is_function_call: true };
            }
        }

        yield {
            data: synthesize ? buffer : answer,
            end_of_stream: true,
            latency: latencyData,
            is_function_call: false,
        };

        this.startedStreaming = false;
    }

    // ------------------------------------------------------------------
    // Chat completions — non-streaming
    // ------------------------------------------------------------------

    private async _generateChat(
        messages: Record<string, unknown>[],
        requestJson = false,
        retMetadata = false
    ): Promise<string | [string, Record<string, unknown>]> {
        const responseFormat = this.getResponseFormat(requestJson);
        try {
            const completion = await this.asyncClient.chat.completions.create({
                model: this.model,
                temperature: 0.0,
                messages: messages as unknown as OpenAI.Chat.ChatCompletionMessageParam[],
                stream: false,
                response_format: responseFormat as OpenAI.Chat.Completions.ChatCompletionCreateParams["response_format"],
            });

            const res = completion?.choices?.[0]?.message.content ?? "";

            if (retMetadata) {
                return [
                    res,
                    {
                        llm_host: this.llmHost ?? null,
                        service_tier:
                            "service_tier" in completion
                                ? (completion.service_tier as string)
                                : null,
                    },
                ];
            }
            return res;
        } catch (e) {
            this._handleOpenAIError(e);
            throw e;
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private getResponseFormat(
        isJsonFormat: boolean
    ): { type: "json_object" } | { type: "text" } {
        return isJsonFormat && JSON_MODE_MODELS.has(this.model)
            ? { type: "json_object" }
            : { type: "text" };
    }

    get getModel(): string {
        return this.model
    }

    private _handleOpenAIError(e: unknown): void {
        if (e instanceof AuthenticationError) {
            logger.error(`OpenAI authentication failed: Invalid or expired API key - ${e}`);
        } else if (e instanceof PermissionDeniedError) {
            logger.error(`OpenAI permission denied (403): ${e}`);
        } else if (e instanceof NotFoundError) {
            logger.error(`OpenAI resource not found (404): Check model name or endpoint - ${e}`);
        } else if (e instanceof RateLimitError) {
            logger.error(`OpenAI rate limit exceeded: ${e}`);
        } else if (e instanceof APIConnectionError) {
            logger.error(`OpenAI connection error: ${e}`);
        } else if (e instanceof APIError) {
            logger.error(`OpenAI API error: ${e}`);
        } else {
            logger.error(`OpenAI unexpected error: ${e}`);
        }
    }
}