import { configureLogger } from "../helper/logger";
import { BaseAgent } from "./base";
import { OpenAiLLM } from "../llms/openai.llm";
import { formatMessages } from "../helper/utils";
import { CHECK_FOR_COMPLETION_PROMPT, VOICEMAIL_DETECTION_PROMPT } from "../prompts";
import type { LLMStreamChunk } from "../llms/types";

const logger = configureLogger("streamingContextualAgent");

export class StreamingContextualAgent extends BaseAgent {
    private llm: OpenAiLLM;
    private conversationCompletionLlm: OpenAiLLM;
    private voicemailLlm: OpenAiLLM;
    protected history: Record<string, unknown>[];

    constructor(llm: OpenAiLLM) {
        super();
        this.llm = llm;
        this.conversationCompletionLlm = new OpenAiLLM(
            100,
            40,
            process.env.CHECK_FOR_COMPLETION_LLM ?? llm.getModel
        );
        this.voicemailLlm = new OpenAiLLM(
            100,
            40,
            process.env.VOICEMAIL_DETECTION_LLM ?? "gpt-4.1-mini"
        );
        this.history = [{ content: "" }];
    }

    // ------------------------------------------------------------------
    // Completion check
    // ------------------------------------------------------------------

    async checkForCompletion(
        messages: Record<string, unknown>[],
        checkForCompletionPrompt: string = CHECK_FOR_COMPLETION_PROMPT
    ): Promise<[Record<string, unknown>, Record<string, unknown>]> {
        try {
            const prompt = [
                { role: "system", content: checkForCompletionPrompt },
                { role: "user", content: formatMessages(messages as never) },
            ];

            const startTime = Date.now();
            const [response, metadata] = (await this.conversationCompletionLlm.generate(
                prompt as never,
                false,
                true
            )) as [string, Record<string, unknown>];
            const latencyMs = Date.now() - startTime;

            const hangup = JSON.parse(response) as Record<string, unknown>;
            metadata["latency_ms"] = latencyMs;

            return [hangup, metadata];
        } catch (e) {
            logger.error(`check_for_completion exception: ${e}`);
            return [{ hangup: "No" }, {}];
        }
    }

    // ------------------------------------------------------------------
    // Voicemail detection
    // ------------------------------------------------------------------

    async checkForVoicemail(
        userMessage: string,
        voicemailDetectionPrompt?: string
    ): Promise<[Record<string, unknown>, Record<string, unknown>]> {
        try {
            const detectionPrompt = voicemailDetectionPrompt ?? VOICEMAIL_DETECTION_PROMPT;
            const prompt = [
                { role: "system", content: detectionPrompt },
                { role: "user", content: `User message: ${userMessage}` },
            ];

            const startTime = Date.now();
            const [response, metadata] = (await this.voicemailLlm.generate(
                prompt as never,
                false,
                true
            )) as [string, Record<string, unknown>];
            const latencyMs = Date.now() - startTime;

            const result = JSON.parse(response) as Record<string, unknown>;
            metadata["latency_ms"] = latencyMs;

            return [result, metadata];
        } catch (e) {
            logger.error(`check_for_voicemail exception: ${e}`);
            return [{ is_voicemail: "No" }, {}];
        }
    }

    // ------------------------------------------------------------------
    // Generate
    // ------------------------------------------------------------------

    async *generate(
        history: Record<string, unknown>[],
        synthesize = false,
        metaInfo: boolean | undefined
    ): AsyncGenerator<LLMStreamChunk> {
        yield* this.llm.generateStream(history as never, synthesize, metaInfo);
    }
}