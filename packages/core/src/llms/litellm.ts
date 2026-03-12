import { configureLogger } from "../helper/logger";
import { DEFAULT_LANGUAGE_CODE } from "../constants";
import { convertToRequestLog, nowMs } from "../helper/utils";
import { BaseLLM } from "./llm";
import { ToolCallAccumulator } from "./toolcall.accumulator";
import type { LLMStreamChunk, LatencyData } from "./types";

// Suppress LiteLLM-equivalent noise — handled by your logger config

const logger = configureLogger("liteLLM");

// LiteLLM JS package: https://www.npmjs.com/package/litellm
import { completion } from "litellm";

interface LiteLLMOptions {
    llm_key?: string;
    base_url?: string;
    api_version?: string;
    api_tools?: {
        tools_params: Record<string, Record<string, unknown>>;
        tools: Record<string, unknown>[];
    } | null;
    run_id?: string;
}

export class LiteLLM extends BaseLLM {
    protected model: string;
    protected language: string;
    protected modelArgs: Record<string, unknown>;
    protected triggerFunctionCall: boolean;
    protected tools: Record<string, unknown>[] | string;
    protected apiParams: Record<string, Record<string, unknown>>;
    protected runId: string;
    protected startedStreaming = false;

    constructor(
        model: string,
        maxTokens = 30,
        bufferSize = 40,
        temperature = 0.0,
        language = DEFAULT_LANGUAGE_CODE,
        kwargs: LiteLLMOptions = {}
    ) {
        super(maxTokens, bufferSize);
        this.model = model;
        this.language = language;
        this.runId = kwargs.run_id ?? "";

        const apiKey = kwargs.llm_key ?? process.env.LITELLM_MODEL_API_KEY;
        const apiBase = kwargs.base_url ?? process.env.LITELLM_MODEL_API_BASE;
        const apiVersion = kwargs.api_version ?? process.env.LITELLM_MODEL_API_VERSION;

        this.modelArgs = { max_tokens: maxTokens, temperature, model };
        if (apiKey) this.modelArgs["api_key"] = apiKey;
        if (apiBase) this.modelArgs["api_base"] = apiBase;
        if (apiVersion) this.modelArgs["api_version"] = apiVersion;

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
    }

    // ------------------------------------------------------------------
    // Streaming
    // ------------------------------------------------------------------

    async *generateStream(
        messages: Record<string, unknown>[],
        synthesize = true,
        metaInfo: Record<string, unknown> | null = null,
        toolChoice?: string
    ): AsyncGenerator<LLMStreamChunk> {
        if (!messages.length) throw new Error("No messages provided");

        let answer = "";
        let buffer = "";
        let firstTokenTime: number | null = null;

        const modelArgs: Record<string, unknown> = {
            ...this.modelArgs,
            messages,
            stream: true,
            stop: ["User:"],
        };

        const parsedTools =
            typeof this.tools === "string"
                ? (JSON.parse(this.tools) as Record<string, unknown>[])
                : this.tools;

        if (this.triggerFunctionCall) {
            modelArgs["tools"] = parsedTools;
            modelArgs["tool_choice"] = toolChoice ?? "auto";
            modelArgs["parallel_tool_calls"] = false;
        }

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
        let latencyData: LatencyData = {
            sequence_id: metaInfo ? (metaInfo["sequence_id"] as number) : null,
            first_token_latency_ms: null,
            total_stream_duration_ms: null,
            service_tier: null,
            llm_host: null,
        };

        let completionStream: AsyncIterable<Record<string, unknown>>;

        try {
            completionStream = (await completion(modelArgs as never)) as unknown as AsyncIterable<Record<string, unknown>>;
        } catch (e) {
            this._handleError(e, metaInfo, "stream");
            return;
        }

        for await (const chunk of completionStream) {
            const now = nowMs();

            if (!firstTokenTime) {
                firstTokenTime = now;
                this.startedStreaming = true;
                latencyData = {
                    sequence_id: metaInfo ? (metaInfo["sequence_id"] as number) : null,
                    first_token_latency_ms: firstTokenTime - startTime,
                    total_stream_duration_ms: null,
                    service_tier: null,
                    llm_host: null,
                };
            }

            const choices = chunk["choices"] as Array<Record<string, unknown>>;
            const delta = (choices[0]?.["delta"] ?? {}) as Record<string, unknown>;

            if (delta["tool_calls"] && accumulator) {
                if (buffer) {
                    yield { data: buffer, end_of_stream: true, latency: latencyData, is_function_call: false };
                    buffer = "";
                }

                accumulator.processDelta(
                    delta["tool_calls"] as Array<{
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
            } else if (delta["content"]) {
                if (accumulator) accumulator.receivedTextual = true;
                const content = delta["content"] as string;
                answer += content;
                buffer += content;

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
            const apiCallPayload = accumulator.buildApiPayload(modelArgs, metaInfo ?? {}, answer);
            if (apiCallPayload) {
                yield { data: apiCallPayload, end_of_stream: false, latency: latencyData, is_function_call: true };
            }
        }

        if (synthesize && buffer.trim()) {
            yield { data: buffer, end_of_stream: true, latency: latencyData, is_function_call: false };
        } else if (!synthesize) {
            yield { data: answer, end_of_stream: true, latency: latencyData, is_function_call: false };
        }

        this.startedStreaming = false;
    }

    // ------------------------------------------------------------------
    // Non-streaming
    // ------------------------------------------------------------------

    async generate(
        messages: Record<string, unknown>[],
        stream = false,
        retMetadata = false
    ): Promise<unknown> {
        const modelArgs: Record<string, unknown> = {
            ...this.modelArgs,
            model: this.model,
            messages,
            stream,
        };

        logger.info(`Request to litellm ${JSON.stringify(modelArgs)}`);

        let text = "";
        try {
            const completionResult = (await completion(modelArgs as never)) as unknown as Record<string, unknown>;
            const choices = completionResult["choices"] as Array<Record<string, unknown>>;
            const message = choices[0]?.["message"] as Record<string, unknown>;
            text = (message?.["content"] as string) ?? "";
        } catch (e) {
            const shouldRethrow = this._handleError(e, null, "generate");
            if (shouldRethrow) throw e;
        }

        return retMetadata ? [text, {}] : text;
    }

    // ------------------------------------------------------------------
    // Error handling
    // ------------------------------------------------------------------

    /** Returns true if the error should be re-thrown. */
    private _handleError(
        e: unknown,
        metaInfo: Record<string, unknown> | null,
        context: "stream" | "generate"
    ): boolean {
        const message = e instanceof Error ? e.message : String(e);

        // Content policy — log and degrade gracefully, don't rethrow
        if (message.toLowerCase().includes("content policy") || message.toLowerCase().includes("content_policy")) {
            logger.error(`Content policy violation in ${context}: ${message}`);
            if (metaInfo && this.runId) {
                convertToRequestLog({
                    message: `Content Policy Violation: ${message}`,
                    metaInfo: metaInfo as never,
                    model: this.model,
                    component: "llm",
                    direction: "error",
                    isCached: false,
                    runId: this.runId,
                });
            }
            return false;
        }

        if (message.toLowerCase().includes("authentication") || message.toLowerCase().includes("api key")) {
            logger.error(`LiteLLM authentication failed: Invalid or expired API key - ${e}`);
        } else if (message.toLowerCase().includes("rate limit")) {
            logger.error(`LiteLLM rate limit exceeded: ${e}`);
        } else if (message.toLowerCase().includes("connection")) {
            logger.error(`LiteLLM connection error: ${e}`);
        } else {
            logger.error(`LiteLLM unexpected error in ${context}: ${message}`);
        }

        return true;
    }
}