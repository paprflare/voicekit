import { configureLogger } from "../helper/logger";
import { BaseAgent } from "./base";
import { RAGServiceClientSingleton, type RAGContext } from "../helper/rag.service.client";
import { nowMs } from "../helper/utils";
import type { LLMStreamChunk } from "../llms/types";
import { SUPPORTED_LLM_PROVIDERS } from "../providers";
import { OpenAiLLM } from "../llms/openai.llm";
import { VOICEMAIL_DETECTION_PROMPT } from "../prompts";

const logger = configureLogger("knowledgeBaseAgent");

export class KnowledgeBaseAgent extends BaseAgent {
    private config: Record<string, unknown>;
    private agentInformation: string;
    private contextData: Record<string, unknown>;
    private llmModel: string;
    private llm: unknown;
    private conversationCompletionLlm: OpenAiLLM;
    private voicemailLlm: OpenAiLLM;
    private ragConfig: Record<string, unknown>;
    private ragServerUrl: string;

    constructor(config: Record<string, unknown>) {
        super();
        this.config = config;
        this.agentInformation = (config["agent_information"] as string | undefined) ?? "Knowledge-based AI assistant";
        this.contextData = (config["context_data"] as Record<string, unknown> | undefined) ?? {};
        this.llmModel = (config["model"] as string | undefined) ?? "gpt-4o";

        this.llm = this.initializeLlm();

        this.conversationCompletionLlm = new OpenAiLLM(process.env.CHECK_FOR_COMPLETION_LLM ?? this.llmModel);
        this.voicemailLlm = new OpenAiLLM(process.env.VOICEMAIL_DETECTION_LLM ?? "gpt-4.1-mini",);

        this.ragConfig = this.initializeRagConfig();
        this.ragServerUrl = process.env.RAG_SERVER_URL ?? "http://localhost:8000";

        logger.info(`KnowledgeBaseAgent initialized with RAG collections: ${JSON.stringify(this.ragConfig["collections"] ?? [])}`);
    }

    // ------------------------------------------------------------------
    // LLM initializer
    // ------------------------------------------------------------------

    private initializeLlm(): unknown {
        try {
            let provider =
                (this.config["provider"] as string | undefined) ??
                (this.config["llm_provider"] as string | undefined) ??
                "openai";

            if (!(provider in SUPPORTED_LLM_PROVIDERS)) {
                logger.warn(`Unknown provider: ${provider}, using openai`);
                provider = "openai";
            }

            const llmKwargs: Record<string, unknown> = {
                model: this.llmModel,
                temperature: (this.config["temperature"] as number | undefined) ?? 0.7,
                max_tokens: (this.config["max_tokens"] as number | undefined) ?? 150,
                provider,
            };

            for (const key of [
                "llm_key", "base_url", "api_version", "language",
                "api_tools", "buffer_size", "reasoning_effort", "service_tier",
            ]) {
                if (this.config[key] != null) llmKwargs[key] = this.config[key];
            }

            const LlmClass = SUPPORTED_LLM_PROVIDERS[provider as keyof typeof SUPPORTED_LLM_PROVIDERS];
            return new LlmClass(JSON.stringify(llmKwargs));
        } catch (e) {
            logger.error(`Failed to create LLM: ${e}, falling back to basic OpenAI`);
            return new OpenAiLLM(this.llmModel);
        }
    }

    // ------------------------------------------------------------------
    // RAG config initializer
    // ------------------------------------------------------------------

    private initializeRagConfig(): Record<string, unknown> {
        const ragConfig = (this.config["rag_config"] as Record<string, unknown> | undefined) ?? {};

        if (!Object.keys(ragConfig).length) {
            logger.warn("No RAG config provided");
            return {};
        }

        const collections: string[] = [];
        const usedSources = ragConfig["used_sources"] as Record<string, unknown>[] | null | undefined;

        if ("vector_store" in ragConfig) {
            const vectorStore = ragConfig["vector_store"] as Record<string, unknown>;
            const providerConfig = (vectorStore["provider_config"] as Record<string, unknown>) ?? {};

            if (usedSources?.length) {
                for (const source of usedSources) {
                    const vectorId = source["vector_id"] as string | undefined;
                    if (vectorId) collections.push(vectorId);
                }
            } else {
                const vectorIds = providerConfig["vector_ids"];
                if (Array.isArray(vectorIds)) {
                    collections.push(...(vectorIds as string[]));
                } else if (providerConfig["vector_id"]) {
                    collections.push(providerConfig["vector_id"] as string);
                } else {
                    logger.error("No vector_id or vector_ids found in rag_config");
                }
            }
        } else {
            logger.error("No vector_store in rag_config");
        }

        return {
            collections,
            similarity_top_k: (ragConfig["similarity_top_k"] as number | undefined) ?? 10,
            used_sources: usedSources ?? [],
        };
    }

    // ------------------------------------------------------------------
    // check_for_completion
    // ------------------------------------------------------------------

    async checkForCompletion(
        messages: Record<string, unknown>[],
        checkForCompletionPrompt: string
    ): Promise<[Record<string, unknown>, Record<string, unknown>]> {
        try {
            const prompt = [
                { role: "system", content: checkForCompletionPrompt },
                ...messages.map(message => ({ role: "user", content: message.content })),
            ];
            const startTime = Date.now();
            const [response, metadata] = await (this.conversationCompletionLlm as OpenAiLLM).generate(prompt, true, true) as [string, Record<string, unknown>];
            const latencyMs = Date.now() - startTime;
            const result = JSON.parse(response as string) as Record<string, unknown>;
            (metadata as Record<string, unknown>)["latency_ms"] = latencyMs;
            return [result, metadata as Record<string, unknown>];
        } catch (e) {
            logger.error(`check_for_completion error: ${e}`);
            return [{ hangup: "No" }, {}];
        }
    }

    // ------------------------------------------------------------------
    // check_for_voicemail
    // ------------------------------------------------------------------

    async checkForVoicemail(
        userMessage: string,
        voicemailDetectionPrompt?: string
    ): Promise<[Record<string, unknown>, Record<string, unknown>]> {
        try {
            const detectionPrompt = voicemailDetectionPrompt ?? VOICEMAIL_DETECTION_PROMPT;
            const prompt = [
                {
                    role: "system",
                    content: detectionPrompt + `
            Respond only in this JSON format:
            {
              "is_voicemail": "Yes" or "No"
            }
          `,
                },
                { role: "user", content: `User message: ${userMessage}` },
            ];

            const startTime = Date.now();
            const [response, metadata] = await (this.voicemailLlm as OpenAiLLM).generate(prompt, true, true) as [string, Record<string, unknown>];
            const latencyMs = Date.now() - startTime;
            const result = JSON.parse(response as string) as Record<string, unknown>;
            (metadata as Record<string, unknown>)["latency_ms"] = latencyMs;
            return [result, metadata as Record<string, unknown>];
        } catch (e) {
            logger.error(`check_for_voicemail exception: ${e}`);
            return [{ is_voicemail: "No" }, {}];
        }
    }

    // ------------------------------------------------------------------
    // _add_rag_context
    // ------------------------------------------------------------------

    private async addRagContext(
        messages: Record<string, unknown>[]
    ): Promise<[Record<string, unknown>[], Record<string, unknown>]> {
        const collections = this.ragConfig["collections"] as string[] | undefined;

        if (!collections?.length) {
            return [messages, { status: "error", message: "No knowledgebases configured" }];
        }

        try {
            const client = await RAGServiceClientSingleton.getClient(this.ragServerUrl);
            const latestMessage = messages.length
                ? (messages[messages.length - 1]!["content"] as string)
                : "";

            const ragResponse = await client.queryForConversation(latestMessage, collections, this.ragConfig["similarity_top_k"] as number, 0.0);
            const ragLatencyData = {
                total_query_time_ms: ragResponse.total_query_time_ms,
                server_processing_time_ms: ragResponse.server_processing_time_ms,
                collections_count: collections.length,
                results_count: ragResponse.total_results,
            };

            if (!ragResponse.contexts?.length) {
                return [messages, {
                    status: "error",
                    message: "No knowledgebase contexts found",
                    latency: ragLatencyData,
                }];
            }

            // Collect which vector IDs were actually used
            const usedVectorIds = new Set<string>();
            for (const context of ragResponse.contexts) {
                const vectorId = context.metadata?.["collection_id"] as string | undefined;
                if (vectorId) usedVectorIds.add(vectorId);
            }

            const usedSources = (this.ragConfig["used_sources"] as Record<string, unknown>[]) ?? [];

            const retrievedSources = usedSources.filter(
                (s) => usedVectorIds.has(s["vector_id"] as string)
            );

            // Build vector_id → source map
            const vectorIdToSource: Record<string, Record<string, unknown>> = {};
            for (const source of usedSources) {
                const vid = source["vector_id"] as string | undefined;
                if (vid) vectorIdToSource[vid] = source;
            }

            // Build per-context entries with latency
            const retrievedContexts = ragResponse.contexts.map(
                (context: RAGContext) => {
                    const vectorId = (context["metadata"] as Record<string, unknown>)?.["collection_id"] as string;
                    const sourceInfo = vectorIdToSource[vectorId] ?? {};
                    return {
                        text: context["text"],
                        score: context["score"],
                        vector_id: sourceInfo["vector_id"],
                        rag_id: sourceInfo["rag_id"],
                        source: sourceInfo["source"],
                    };
                }
            );

            logger.info(
                `RAG: Found ${ragResponse.total_results} contexts, ` +
                `top score: ${((ragResponse.contexts[0] as RAGContext)["score"] as number).toFixed(3)}`
            );

            const ragContext = await client.formatContextForPrompt(ragResponse.contexts);

            // Extract system prompt and remaining messages
            let systemPrompt: string;
            let otherMessages: Record<string, unknown>[];

            if (messages.length && messages[0]!["role"] === "system") {
                systemPrompt = messages[0]!["content"] as string;
                otherMessages = messages.slice(1);
            } else {
                systemPrompt =
                    (this.config["prompt"] as string | undefined) ??
                    `You are ${this.agentInformation}.`;
                otherMessages = messages;
            }

            const enhancedSystemPrompt = `${systemPrompt}

You have access to relevant information from the knowledge base:

${ragContext}

Use this information naturally when it helps answer the user's questions. Don't force references if not relevant to the conversation.`;

            let finalMessages: Record<string, unknown>[] = [
                { role: "system", content: enhancedSystemPrompt },
                ...otherMessages,
            ];

            // Limit history size
            const MAX_MESSAGES = 50;
            if (finalMessages.length > MAX_MESSAGES) {
                finalMessages = [finalMessages[0]!, ...finalMessages.slice(-(MAX_MESSAGES - 1))];
            }

            return [finalMessages, {
                status: "success",
                retrieved_sources: retrievedSources,
                contexts: retrievedContexts,
                latency: ragLatencyData,
            }];
        } catch (e) {
            if ((e as Error).name === "TimeoutError") {
                logger.error("RAG service timeout");
            } else {
                logger.error(`RAG error: ${e}`);
            }
            return [messages, { status: "error", message: "Internal Service Error" }];
        }
    }

    // ------------------------------------------------------------------
    // generate (streaming)
    // ------------------------------------------------------------------

    async *generate(
        message: Record<string, unknown>[],
        kwargs: Record<string, unknown> = {}
    ): AsyncGenerator<Record<string, unknown> | LLMStreamChunk> {
        const metaInfo = kwargs["meta_info"] as Record<string, unknown>;
        const synthesize = (kwargs["synthesize"] as boolean | undefined) ?? true;
        const startTime = nowMs();

        metaInfo["llm_metadata"] = (metaInfo["llm_metadata"] as Record<string, unknown> | undefined) ?? {};
        (metaInfo["llm_metadata"] as Record<string, unknown>)["rag_info"] = {
            all_sources: this.ragConfig["used_sources"] ?? [],
        };

        try {
            const [messagesWithContext, metadata] = await this.addRagContext(message);

            ((metaInfo["llm_metadata"] as Record<string, unknown>)["rag_info"] as Record<string, unknown>)[
                "context_retrieval"
            ] = metadata;

            if (metadata["latency"]) {
                metaInfo["rag_latency"] = {
                    sequence_id: metaInfo["sequence_id"],
                    ...(metadata["latency"] as Record<string, unknown>),
                };
            }

            yield { messages: messagesWithContext };

            const llmWithStream = this.llm as {
                generateStream: (
                    messages: Record<string, unknown>[],
                    opts: { synthesize: boolean; metaInfo: Record<string, unknown> }
                ) => AsyncGenerator<LLMStreamChunk>;
            };

            for await (const chunk of llmWithStream.generateStream(messagesWithContext, {
                synthesize,
                metaInfo,
            })) {
                yield chunk;
            }
        } catch (e) {
            logger.error(`generate() error: ${e}`);
            const chunk: LLMStreamChunk = {
                data: `An error occurred: ${(e as Error).message}`,
                is_function_call: false,
                end_of_stream: true,
                latency: {
                    sequence_id: metaInfo?.["sequence_id"] as number,
                    first_token_latency_ms: 0,
                    total_stream_duration_ms: nowMs() - startTime,
                },
            };

            yield chunk;
        }
    }
}