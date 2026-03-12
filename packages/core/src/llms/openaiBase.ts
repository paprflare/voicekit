import { configureLogger } from "../helper/logger";
import { GPT5_MODEL_PREFIX } from "../constants";
import { ChatRoleSchema, ResponseStreamEventSchema, ResponseItemTypeSchema } from "../enums";
import { convertToRequestLog, computeFunctionPreCallMessage, nowMs } from "../helper/utils";
import { BaseLLM } from "./llm";
import { MessageFormatAdapter } from "./message.models";
import type { LLMStreamChunk, LatencyData, FunctionCallPayload } from "./types";

const logger = configureLogger("openaiCompatibleLlm");

export abstract class OpenAICompatibleLLM extends BaseLLM {
    protected useResponsesApi = false;
    protected previousResponseId: string | null = null;

    // subclass-provided
    protected abstract asyncClient: {
        responses: {
            create: (kwargs: Record<string, unknown>) => Promise<unknown>;
        };
    };
    protected model!: string;
    protected temperature!: number;
    protected runId!: string;
    protected triggerFunctionCall = false;
    protected tools: string | Record<string, unknown>[] = [];
    protected apiParams: Record<string, Record<string, unknown>> = {};
    protected modelArgs: Record<string, unknown> = {};
    protected language: string = "en";
    protected startedStreaming = false;
    protected llmHost?: string;

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------

    protected initResponsesApi(useResponsesApi = false): void {
        this.useResponsesApi = useResponsesApi;
        this.previousResponseId = null;
    }

    protected get responsesClient() {
        return this.asyncClient;
    }

    // ------------------------------------------------------------------
    // Input building
    // ------------------------------------------------------------------

    private buildResponsesInput(
        messages: Record<string, unknown>[]
    ): [string, Record<string, unknown>[]] {
        if (this.previousResponseId) {
            return this.extractNewInput(messages);
        }
        return MessageFormatAdapter.chatToResponsesInput(messages);
    }

    private extractNewInput(
        messages: Record<string, unknown>[]
    ): [string, Record<string, unknown>[]] {
        let instructions = "";
        if (messages.length > 0 && messages[0]?.["role"] === ChatRoleSchema.enum.system) {
            instructions = (messages[0]["content"] as string) ?? "";
        }

        let lastAssistantIdx = -1;
        for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i]?.["role"] === ChatRoleSchema.enum.assistant) {
                lastAssistantIdx = i;
                break;
            }
        }

        if (lastAssistantIdx < 0) {
            const [, inputItems] = MessageFormatAdapter.chatToResponsesInput(messages);
            return [instructions, inputItems];
        }

        const newMessages = messages.slice(lastAssistantIdx + 1);
        const [, inputItems] = MessageFormatAdapter.chatToResponsesInput(newMessages);
        return [instructions, inputItems];
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private isStaleResponseError(error: unknown): boolean {
        // HTTP 400 from OpenAI SDK surfaces as an error with status 400
        return (
            typeof error === "object" &&
            error !== null &&
            "status" in error &&
            (error as { status: number }).status === 400
        );
    }

    private parseTools(): Record<string, unknown>[] {
        if (!this.triggerFunctionCall) return [];
        return typeof this.tools === "string"
            ? (JSON.parse(this.tools) as Record<string, unknown>[])
            : this.tools;
    }

    override invalidateResponseChain(): void {
        this.previousResponseId = null;
    }

    // ------------------------------------------------------------------
    // Streaming (async generator)
    // ------------------------------------------------------------------

    async *generateStreamResponses(
        messages: Record<string, unknown>[],
        synthesize = true,
        requestJson = false,
        metaInfo: Record<string, unknown> | undefined = undefined,
        toolChoice?: string
    ): AsyncGenerator<LLMStreamChunk> {
        if (!messages.length) throw new Error("No messages provided");

        const [instructions, inputItems] = this.buildResponsesInput(messages);
        const responsesTools = MessageFormatAdapter.chatToolsToResponsesTools(this.parseTools());

        const createKwargs: Record<string, unknown> = {
            model: this.model,
            instructions: instructions || null,
            input: inputItems,
            stream: true,
            store: true,
            max_output_tokens: this.maxTokens,
            temperature: this.temperature,
            user: metaInfo ? `${this.runId}#${metaInfo["turn_id"]}` : null,
        };

        const serviceTierArg = this.modelArgs["service_tier"] as string | undefined;
        if (serviceTierArg) createKwargs["service_tier"] = serviceTierArg;

        if (this.model.startsWith(GPT5_MODEL_PREFIX)) {
            const reasoningEffort = this.modelArgs["reasoning_effort"];
            if (reasoningEffort) createKwargs["reasoning"] = { effort: reasoningEffort };
        }

        if (this.previousResponseId) {
            createKwargs["previous_response_id"] = this.previousResponseId;
        }

        if (responsesTools.length) {
            createKwargs["tools"] = responsesTools;
            createKwargs["tool_choice"] = toolChoice ?? "auto";
            createKwargs["parallel_tool_calls"] = false;
        }

        if (requestJson) {
            createKwargs["text"] = { format: { type: "json_object" } };
        }

        let answer = "";
        let buffer: string = "";
        const funcCallArgs: Record<string, string> = {};
        const funcCallNames: Record<string, string> = {};
        const funcCallIds: Record<string, string> = {};
        let gavePreCallMsg = false;
        let receivedTextual = false;

        const startTime = nowMs();
        let firstTokenTime: number | null = null;
        let latencyData: LatencyData | null = null;
        let serviceTier: string | null = null;
        const llmHost = this.llmHost ?? null;

        // --- open stream ---
        let stream: AsyncIterable<Record<string, unknown>>;
        try {
            stream = (await this.responsesClient.responses.create(createKwargs)) as AsyncIterable<Record<string, unknown>>;
        } catch (e) {
            if (this.previousResponseId && this.isStaleResponseError(e)) {
                logger.warn(`Stale previous_response_id, retrying with full history: ${e}`);
                this.previousResponseId = null;
                yield* this.generateStreamResponses(messages, synthesize, requestJson, metaInfo, toolChoice);
                return;
            }
            logger.error(`Responses API error: ${e}`);
            throw e;
        }

        // --- consume events ---
        for await (const event of stream) {
            const now = nowMs();
            const eventType = event["type"] as string;

            if (eventType === ResponseStreamEventSchema.enum["response.created"]) {
                const response = event["response"] as Record<string, unknown>;
                this.previousResponseId = response["id"] as string;
                serviceTier = (response["service_tier"] as string) ?? null;
                continue;
            }

            if (eventType === ResponseStreamEventSchema.enum["response.failed"]) {
                const response = event["response"] as Record<string, unknown>;
                const errorInfo = response["error"] ?? response["last_error"];
                logger.error(`Responses API stream failed: ${errorInfo}`);
                this.previousResponseId = null;
                throw new Error(`Response failed: ${JSON.stringify(errorInfo)}`);
            }

            if (eventType === ResponseStreamEventSchema.enum["response.incomplete"]) {
                logger.warn("Responses API stream incomplete, partial response returned");
                this.previousResponseId = null;
                break;
            }

            const isFirstTokenEvent =
                eventType === ResponseStreamEventSchema.enum["response.output_text.delta"] ||
                eventType === ResponseStreamEventSchema.enum["response.function_call_arguments.delta"];

            if (!firstTokenTime && isFirstTokenEvent) {
                firstTokenTime = now;
                this.startedStreaming = true;
                latencyData = {
                    sequence_id: metaInfo ? (metaInfo["sequence_id"] as number) : null,
                    first_token_latency_ms: firstTokenTime - startTime,
                    total_stream_duration_ms: null,
                    service_tier: serviceTier,
                    llm_host: llmHost,
                };
            }

            if (eventType === ResponseStreamEventSchema.enum["response.output_text.delta"]) {
                receivedTextual = true;
                const delta = event["delta"] as string;
                answer += delta;
                buffer += delta;

                if (synthesize && buffer.length >= this.bufferSize) {
                    const parts = buffer.split(/ (?=[^ ]*$)/); // rsplit on last space
                    if (buffer as string) {
                        yield { data: parts[0], end_of_stream: true, latency: latencyData, is_function_call: false };
                        buffer = ""; // Reset buffer to an empty string
                    }
                }

            } else if (eventType === ResponseStreamEventSchema.enum["response.output_item.added"]) {
                const item = event["item"] as Record<string, unknown>;

                if (item["type"] === ResponseItemTypeSchema.enum.function_call) {
                    if (buffer) {
                        yield { data: buffer, end_of_stream: true, latency: latencyData, is_function_call: false };
                        buffer = "";
                    }

                    const itemId = item["id"] as string;
                    funcCallArgs[itemId] = "";
                    funcCallNames[itemId] = item["name"] as string;
                    funcCallIds[itemId] = item["call_id"] as string;

                    if (!gavePreCallMsg && !receivedTextual && this.triggerFunctionCall) {
                        gavePreCallMsg = true;
                        const funcName = item["name"] as string;
                        const apiToolPreCallMessage =
                            (this.apiParams[funcName]?.["pre_call_message"] as string | Record<string, string> | null) ?? null;
                        const detectedLang = metaInfo ? (metaInfo["detected_language"] as string) : null;
                        const activeLanguage = detectedLang ?? this.language;
                        const preMsg = computeFunctionPreCallMessage(activeLanguage, funcName, apiToolPreCallMessage);
                        if (preMsg) {
                            yield {
                                data: preMsg,
                                end_of_stream: true,
                                latency: latencyData,
                                is_function_call: false,
                                function_name: funcName,
                                function_message: apiToolPreCallMessage as string | Record<string, unknown> | null,
                            };
                        }
                    }
                }

            } else if (eventType === ResponseStreamEventSchema.enum["response.function_call_arguments.delta"]) {
                const itemId = event["item_id"] as string;
                funcCallArgs[itemId] = (funcCallArgs[itemId] ?? "") + (event["delta"] as string);

            } else if (eventType === ResponseStreamEventSchema.enum["response.completed"]) {
                const response = event["response"] as Record<string, unknown>;
                if (response["id"]) this.previousResponseId = response["id"] as string;
                serviceTier = serviceTier ?? ((response["service_tier"] as string) ?? null);
                break;
            }
        }

        if (latencyData) {
            latencyData.total_stream_duration_ms = nowMs() - startTime;
            if (serviceTier) latencyData.service_tier = serviceTier;
        }

        // --- function call handling ---
        if (Object.keys(funcCallArgs).length && this.triggerFunctionCall) {
            const firstItemId = Object.keys(funcCallArgs)[0]!;
            const funcName = funcCallNames[firstItemId]!;
            const callId = funcCallIds[firstItemId];
            const argumentsStr = funcCallArgs[firstItemId];

            if (funcName in this.apiParams) {
                const funcConf = this.apiParams[funcName]!;
                logger.info(`Payload to send ${argumentsStr} func_dict ${JSON.stringify(funcConf)}`);

                const method = funcConf["method"] as string | undefined;
                const apiCallPayload: FunctionCallPayload = {
                    url: funcConf["url"] as string ?? null,
                    method: method ? method.toLowerCase() : null,
                    param: funcConf["param"] ?? null,
                    api_token: funcConf["api_token"] as string ?? null,
                    headers: funcConf["headers"] as Record<string, unknown> ?? null,
                    model_args: createKwargs,
                    meta_info: metaInfo ?? {},
                    called_fun: funcName,
                    model_response: [{
                        index: 0,
                        id: callId,
                        function: { name: funcName, arguments: argumentsStr },
                        type: "function",
                    }],
                    tool_call_id: callId as string,
                    textual_response: receivedTextual ? answer.trim() : null,
                    resp: null,
                };

                const toolSpec = responsesTools.find(
                    (t) => (t as Record<string, unknown>)["name"] === funcName
                ) as Record<string, unknown> | undefined;

                if (toolSpec) {
                    try {
                        const parsedArgs = JSON.parse(argumentsStr as string) as Record<string, unknown>;
                        const params = toolSpec["parameters"] as Record<string, unknown> | undefined;
                        const requiredKeys = (params?.["required"] as string[]) ?? [];

                        if (params && requiredKeys.every((k) => k in parsedArgs)) {
                            convertToRequestLog({
                                message: argumentsStr,
                                metaInfo: metaInfo as never,
                                model: this.model,
                                component: "llm",
                                direction: "response",
                                isCached: false,
                                runId: this.runId,
                            });
                            for (const [k, v] of Object.entries(parsedArgs)) {
                                (apiCallPayload as Record<string, unknown>)[k] = v;
                            }
                        } else {
                            apiCallPayload.resp = null;
                        }
                    } catch (e) {
                        logger.error(`Error parsing function arguments: ${e}`);
                        apiCallPayload.resp = null;
                    }
                } else {
                    apiCallPayload.resp = null;
                }

                yield {
                    data: apiCallPayload,
                    end_of_stream: false,
                    latency: latencyData,
                    is_function_call: true,
                };
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
    // Non-streaming
    // ------------------------------------------------------------------

    async generateResponses(
        messages: Record<string, unknown>[],
        requestJson = false,
        retMetadata = false
    ): Promise<string | [string, Record<string, unknown>]> {
        const [instructions, inputItems] = this.buildResponsesInput(messages);

        const createKwargs: Record<string, unknown> = {
            model: this.model,
            instructions: instructions || null,
            input: inputItems,
            store: true,
            max_output_tokens: this.maxTokens,
            temperature: 0.0,
        };

        const serviceTier = this.modelArgs["service_tier"] as string | undefined;
        if (serviceTier) createKwargs["service_tier"] = serviceTier;

        if (this.model.startsWith(GPT5_MODEL_PREFIX)) {
            const reasoningEffort = this.modelArgs["reasoning_effort"];
            if (reasoningEffort) createKwargs["reasoning"] = { effort: reasoningEffort };
        }

        if (this.previousResponseId) {
            createKwargs["previous_response_id"] = this.previousResponseId;
        }

        if (requestJson) {
            createKwargs["text"] = { format: { type: "json_object" } };
        }

        const llmHost = this.llmHost ?? null;

        try {
            const response = (await this.responsesClient.responses.create(createKwargs)) as Record<string, unknown>;
            this.previousResponseId = response["id"] as string;
            const res = response["output_text"] as string;

            if (retMetadata) {
                const metadata = {
                    llm_host: llmHost,
                    service_tier: (response["service_tier"] as string) ?? null,
                };
                return [res, metadata];
            }
            return res;
        } catch (e) {
            if (this.previousResponseId && this.isStaleResponseError(e)) {
                logger.warn(`Stale previous_response_id, retrying with full history: ${e}`);
                this.previousResponseId = null;
                return this.generateResponses(messages, requestJson, retMetadata);
            }
            logger.error(`Responses API error: ${e}`);
            throw e;
        }
    }
    async generate(
        messages: Record<string, unknown>[],
        stream?: boolean,
        retMetadata?: boolean
    ): Promise<unknown> {
        return this.generateResponses(messages, false, retMetadata ?? false);
    }
}