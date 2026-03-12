import { configureLogger } from "../helper/logger";
import { convertToRequestLog, computeFunctionPreCallMessage } from "../helper/utils";
import type { FunctionCallPayload } from "./types";

const logger = configureLogger("toolCallAccumulator");

interface ToolCallDelta {
    index: number;
    id?: string;
    function: {
        name?: string;
        arguments?: string;
    };
}

interface AccumulatedToolCall {
    index: number;
    id: string;
    function: {
        name: string;
        arguments: string;
    };
    type: "function";
}

type PreCallMessageResult = [message: string, functionName: string, rawConfig: string | Record<string, string> | null];

/**
 * Accumulates streamed tool-call deltas into complete function-call payloads.
 *
 * Used by OpenAI, Azure, and LiteLLM providers to consolidate the identical
 * tool-call streaming logic that was previously duplicated across each provider.
 */
export class ToolCallAccumulator {
    private apiParams: Record<string, Record<string, unknown>>;
    private tools: Record<string, unknown>[];
    private language: string;
    private model: string;
    private runId: string;
    private finalToolCalls: Record<number, AccumulatedToolCall> = {};
    private calledFun: string | null = null;
    private gavePreCallMsg = false;
    receivedTextual = false;

    constructor(opts: {
        apiParams: Record<string, Record<string, unknown>>;
        tools: Record<string, unknown>[];
        language: string;
        model: string;
        runId: string;
    }) {
        this.apiParams = opts.apiParams;
        this.tools = opts.tools;
        this.language = opts.language;
        this.model = opts.model;
        this.runId = opts.runId;
    }

    // ------------------------------------------------------------------
    // Accumulation
    // ------------------------------------------------------------------

    /** Accumulate streamed tool-call chunks into `finalToolCalls`. */
    processDelta(toolCallsDeltas: ToolCallDelta[] | null | undefined): void {
        for (const toolCall of toolCallsDeltas ?? []) {
            const idx = toolCall.index;
            if (!(idx in this.finalToolCalls)) {
                this.calledFun = toolCall.function.name ?? "";
                logger.info(`Function given by LLM to trigger is - ${this.calledFun}`);
                this.finalToolCalls[idx] = {
                    index: toolCall.index,
                    id: toolCall.id ?? "",
                    function: {
                        name: this.calledFun,
                        arguments: toolCall.function.arguments ?? "",
                    },
                    type: "function",
                };
            } else {
                this.finalToolCalls[idx]!.function.arguments +=
                    toolCall.function.arguments ?? "";
            }
        }
    }

    // ------------------------------------------------------------------
    // Pre-call message
    // ------------------------------------------------------------------

    /**
     * Return a user-facing filler message for the first detected function call.
     * Returns the result once, then null on subsequent calls.
     */
    getPreCallMessage(
        metaInfo: Record<string, unknown> | null
    ): PreCallMessageResult | null {
        if (this.gavePreCallMsg || this.receivedTextual || !this.calledFun) {
            return null;
        }
        this.gavePreCallMsg = true;

        const apiToolPreCallMessage =
            (this.apiParams[this.calledFun]?.["pre_call_message"] as
                | string
                | Record<string, string>
                | null) ?? null;
        const detectedLang = metaInfo
            ? (metaInfo["detected_language"] as string | undefined)
            : null;
        const activeLanguage = detectedLang ?? this.language;
        const preMsg = computeFunctionPreCallMessage(
            activeLanguage,
            this.calledFun,
            apiToolPreCallMessage
        );

        return [preMsg, this.calledFun, apiToolPreCallMessage];
    }

    // ------------------------------------------------------------------
    // Payload building
    // ------------------------------------------------------------------

    /**
     * Build the final API call payload from accumulated tool-call data.
     * Validates required parameters against the tool spec and merges parsed
     * arguments into the payload. Returns null if no tool calls or the
     * function isn't in apiParams.
     */
    buildApiPayload(
        modelArgs: Record<string, unknown>,
        metaInfo: Record<string, unknown>,
        answer: string
    ): FunctionCallPayload | null {
        if (!Object.keys(this.finalToolCalls).length) return null;

        const firstToolCall = this.finalToolCalls[0]!;
        const firstFuncName = firstToolCall.function.name;

        if (!(firstFuncName in this.apiParams)) return null;

        const funcConf = this.apiParams[firstFuncName]!;
        const argumentsReceived = firstToolCall.function.arguments;

        logger.info(
            `Payload to send ${argumentsReceived} func_dict ${JSON.stringify(funcConf)}`
        );
        this.gavePreCallMsg = false;

        const method = funcConf["method"] as string | undefined;
        const apiCallPayload: FunctionCallPayload = {
            url: (funcConf["url"] as string) ?? null,
            method: method ? method.toLowerCase() : null,
            param: funcConf["param"] ?? null,
            api_token: (funcConf["api_token"] as string) ?? null,
            headers: (funcConf["headers"] as Record<string, unknown>) ?? null,
            model_args: modelArgs,
            meta_info: metaInfo,
            called_fun: firstFuncName,
            model_response: Object.values(this.finalToolCalls),
            tool_call_id: firstToolCall.id ?? "",
            textual_response: this.receivedTextual ? answer.trim() : null,
            resp: undefined,
        };

        // Chat Completions tools use nested { function: { name, parameters } }
        const toolSpec = this.tools.find(
            (t) =>
                (t["function"] as Record<string, unknown>)?.["name"] === firstFuncName
        ) as Record<string, unknown> | undefined;

        if (!toolSpec) {
            apiCallPayload.resp = null;
            return apiCallPayload;
        }

        try {
            const parsedArgs = JSON.parse(argumentsReceived) as Record<string, unknown>;
            const funcDef = toolSpec["function"] as Record<string, unknown>;
            const params = funcDef["parameters"] as Record<string, unknown> | undefined;
            const requiredKeys = (params?.["required"] as string[]) ?? [];

            if (params && requiredKeys.every((k) => k in parsedArgs)) {
                convertToRequestLog({
                    message: argumentsReceived,
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

        return apiCallPayload;
    }
}