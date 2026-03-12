import { configureLogger } from "../helper/logger";
import { BaseAgent } from "./base";
import { OpenAiLLM } from "../llms/openai.llm";
import { RAGServiceClientSingleton } from "../helper/rag.service.client";
import { nowMs, formatMessages, updatePromptWithContext } from "../helper/utils";
import type { LLMStreamChunk, LatencyData } from "../llms/types";
import type { GraphAgentConfig } from "../models";
import { VOICEMAIL_DETECTION_PROMPT } from "../prompts";
import { } from "../providers";

// Optional Groq support
let Groq: typeof import("groq").default | null = null;
try {
    Groq = require("groq").default;
} catch {
    Groq = null;
}

const logger = configureLogger("graphAgent");

type RoutingDecision = [
    nextNodeId: string | null,
    extractedParams: Record<string, unknown> | null,
    latencyMs: number,
    routingMessages: Record<string, unknown>[] | null,
    routingTools: Record<string, unknown>[] | null
];

export class GraphAgent extends BaseAgent {
    private config: Record<string, unknown>;
    private agentInformation: string;
    private currentNodeId: string;
    private contextData: Record<string, unknown>;
    private llmModel: string;
    private llmKey: string | undefined;
    private baseUrl: string | undefined;
    private openai: InstanceType<typeof import("openai").default>;
    private nodeHistory: string[];
    private currentNodeEntryIndex: number;
    private ragConfigs: Record<string, Record<string, unknown>>;
    private ragServerUrl: string;
    private transitionToolsCache: Map<string, Record<string, unknown>[]>;
    private readonly transitionToolsCacheMaxSize = 100;

    private routingProvider: string;
    private routingModel: string;
    private routingInstructions: string | null;
    private routingReasoningEffort: string | null;
    private routingMaxTokens: number | null;
    private routingClient: unknown;

    private llm: OpenAiLLM;
    private conversationCompletionLlm: OpenAiLLM;
    private voicemailLlm: OpenAiLLM;

    constructor(config: GraphAgentConfig & Record<string, unknown>) {
        super();
        this.config = config as Record<string, unknown>;
        this.agentInformation = (config as Record<string, unknown>)["agent_information"] as string ?? "";
        this.currentNodeId = (config as Record<string, unknown>)["current_node_id"] as string ?? "";
        this.contextData = ((config as Record<string, unknown>)["context_data"] as Record<string, unknown>) ?? {};
        this.llmModel = (config as Record<string, unknown>)["model"] as string ?? "gpt-4o-mini";
        this.llmKey = (config as Record<string, unknown>)["llm_key"] as string | undefined ?? process.env.OPENAI_API_KEY;
        this.baseUrl = (config as Record<string, unknown>)["base_url"] as string | undefined;

        const OpenAI = require("openai").default;
        this.openai = this.baseUrl
            ? new OpenAI({ apiKey: this.llmKey, baseURL: this.baseUrl })
            : new OpenAI({ apiKey: this.llmKey });

        if (this.baseUrl) {
            logger.info(`OpenAI client initialized with custom base_url: ${this.baseUrl}`);
        }

        this.nodeHistory = [this.currentNodeId];
        this.currentNodeEntryIndex = 0;
        this.ragConfigs = this.initializeRagConfigs();
        this.ragServerUrl = process.env.RAG_SERVER_URL ?? "http://localhost:8000";
        this.transitionToolsCache = new Map();

        this.routingProvider = (config as Record<string, unknown>)["routing_provider"] as string ?? "";
        this.routingModel = (config as Record<string, unknown>)["routing_model"] as string ?? "";
        this.routingInstructions = (config as Record<string, unknown>)["routing_instructions"] as string | null ?? null;
        this.routingReasoningEffort = (config as Record<string, unknown>)["routing_reasoning_effort"] as string | null ?? null;
        this.routingMaxTokens = (config as Record<string, unknown>)["routing_max_tokens"] as number | null ?? null;

        logger.info(`GraphAgent routing_instructions loaded: ${!!this.routingInstructions} (length: ${this.routingInstructions?.length ?? 0})`);

        this.initRoutingClient();
        this.llm = this.initializeLlm();

        const llmKwargs: Record<string, unknown> = {};
        if (this.llmKey) llmKwargs["llm_key"] = this.llmKey;
        if (this.baseUrl) llmKwargs["base_url"] = this.baseUrl;

        this.conversationCompletionLlm = new OpenAiLLM(
            100, 40,
            process.env.CHECK_FOR_COMPLETION_LLM ?? this.llmModel ?? "gpt-4o-mini",
            0.1, "en", llmKwargs
        );
        this.voicemailLlm = new OpenAiLLM(
            100, 40,
            process.env.VOICEMAIL_DETECTION_LLM ?? "gpt-4.1-mini",
            0.1, "en", llmKwargs
        );
    }

    // ------------------------------------------------------------------
    // Initialisation helpers
    // ------------------------------------------------------------------

    private initializeLlm(): OpenAiLLM {
        try {
            const provider = (this.config["provider"] ?? this.config["llm_provider"] ?? "openai") as string;
            const resolvedProvider = provider in SUPPORTED_LLM_PROVIDERS ? provider : "openai";
            if (!(provider in SUPPORTED_LLM_PROVIDERS)) {
                logger.warn(`Unknown provider: ${provider}, using openai`);
            }

            const llmKwargs: Record<string, unknown> = {
                model: this.llmModel,
                temperature: this.config["temperature"] ?? 0.7,
                max_tokens: this.config["max_tokens"] ?? 150,
                provider: resolvedProvider,
            };

            for (const key of ["llm_key", "base_url", "api_version", "language", "api_tools", "buffer_size", "reasoning_effort", "service_tier"]) {
                if (this.config[key] != null) llmKwargs[key] = this.config[key];
            }

            const LlmClass = SUPPORTED_LLM_PROVIDERS[resolvedProvider];
            return new LlmClass(llmKwargs) as OpenAiLLM;
        } catch (e) {
            logger.error(`Failed to create LLM: ${e}, falling back to default OpenAiLLM`);
            return new OpenAiLLM(100, 40, this.llmModel ?? "gpt-4o-mini", 0.1, "en", {
                llm_key: this.llmKey ?? process.env.OPENAI_API_KEY,
            });
        }
    }

    private initializeRagConfigs(): Record<string, Record<string, unknown>> {
        const ragConfigs: Record<string, Record<string, unknown>> = {};
        const nodes = (this.config["nodes"] as Record<string, unknown>[]) ?? [];

        for (const node of nodes) {
            const ragConfig = node["rag_config"] as Record<string, unknown> | undefined;
            if (!ragConfig) continue;

            const collections: string[] = [];

            if ("vector_id" in ragConfig) {
                collections.push(ragConfig["vector_id"] as string);
            } else if (
                "provider_config" in ragConfig &&
                typeof ragConfig["provider_config"] === "object" &&
                (ragConfig["provider_config"] as Record<string, unknown>)["vector_id"]
            ) {
                collections.push((ragConfig["provider_config"] as Record<string, unknown>)["vector_id"] as string);
            } else {
                try {
                    const vectorStore = (ragConfig["vector_store"] ?? {}) as Record<string, unknown>;
                    const providerConfig = (vectorStore["provider_config"] ?? {}) as Record<string, unknown>;
                    if (providerConfig["vector_id"]) {
                        collections.push(providerConfig["vector_id"] as string);
                    }
                } catch { /* ignore */ }
            }

            ragConfigs[node["id"] as string] = {
                collections,
                similarity_top_k: ragConfig["similarity_top_k"] ?? 10,
                temperature: ragConfig["temperature"] ?? 0.7,
                model: ragConfig["model"] ?? "gpt-4o",
                max_tokens: ragConfig["max_tokens"] ?? 150,
            };

            logger.info(`Initialized RAG config for node ${node["id"]} with collections: ${collections}`);
        }
        return ragConfigs;
    }

    private initRoutingClient(): void {
        const groqKey = process.env.GROQ_API_KEY;
        const groqAvailable = !!Groq && !!groqKey;

        if (!this.routingProvider) {
            this.routingProvider = groqAvailable ? "groq" : "openai";
        }

        if (this.routingProvider === "groq") {
            if (groqAvailable) {
                this.routingClient = new Groq!({ apiKey: groqKey });
                if (!this.routingModel) {
                    this.routingModel = process.env.DEFAULT_ROUTING_MODEL_GROQ ?? "llama-3.3-70b-versatile";
                }
                logger.info(`Routing initialized with Groq (${this.routingModel}) - fast mode ~200ms`);
            } else {
                logger.warn("Groq requested but unavailable, falling back to OpenAI");
                this.routingClient = this.openai;
                this.routingProvider = "openai";
                this.routingModel = process.env.DEFAULT_ROUTING_MODEL_OPENAI ?? "gpt-4.1-mini";
            }
        } else {
            this.routingClient = this.openai;
            if (!this.routingModel) {
                this.routingModel = process.env.DEFAULT_ROUTING_MODEL_OPENAI ?? "gpt-4.1-mini";
            }
            logger.info(`Routing initialized with OpenAI (${this.routingModel})`);
        }
    }

    // ------------------------------------------------------------------
    // Completion / voicemail checks
    // ------------------------------------------------------------------

    async checkForCompletion(
        messages: Record<string, unknown>[],
        checkForCompletionPrompt: string
    ): Promise<[Record<string, unknown>, Record<string, unknown>]> {
        try {
            const prompt = [
                { role: "system", content: checkForCompletionPrompt },
                { role: "user", content: formatMessages(messages as never) },
            ];
            const startTime = Date.now();
            const [response, metadata] = (await this.conversationCompletionLlm.generate(
                prompt as never, false, true
            )) as [string, Record<string, unknown>];
            const hangup = JSON.parse(response) as Record<string, unknown>;
            metadata["latency_ms"] = Date.now() - startTime;
            return [hangup, metadata];
        } catch (e) {
            logger.error(`check_for_completion exception: ${e}`);
            return [{ hangup: "No" }, {}];
        }
    }

    async checkForVoicemail(
        userMessage: string,
        voicemailDetectionPrompt?: string
    ): Promise<[Record<string, unknown>, Record<string, unknown>]> {
        try {
            const detectionPrompt = voicemailDetectionPrompt ?? VOICEMAIL_DETECTION_PROMPT;
            const prompt = [
                {
                    role: "system",
                    content: `${detectionPrompt}\n\nRespond only in this JSON format:\n{\n  "is_voicemail": "Yes" or "No"\n}`,
                },
                { role: "user", content: `User message: ${userMessage}` },
            ];
            const startTime = Date.now();
            const [response, metadata] = (await this.voicemailLlm.generate(
                prompt as never, false, true
            )) as [string, Record<string, unknown>];
            const result = JSON.parse(response) as Record<string, unknown>;
            metadata["latency_ms"] = Date.now() - startTime;
            return [result, metadata];
        } catch (e) {
            logger.error(`check_for_voicemail exception: ${e}`);
            return [{ is_voicemail: "No" }, {}];
        }
    }

    // ------------------------------------------------------------------
    // Routing tools
    // ------------------------------------------------------------------

    private buildTransitionTools(node: Record<string, unknown>): Record<string, unknown>[] {
        const nodeId = node["id"] as string | undefined;
        if (nodeId && this.transitionToolsCache.has(nodeId)) {
            return this.transitionToolsCache.get(nodeId)!;
        }

        const edges = (node["edges"] as Record<string, unknown>[]) ?? [];
        const tools: Record<string, unknown>[] = edges.map((edge) => {
            const toNodeId = edge["to_node_id"] as string;
            const funcName = (edge["function_name"] as string) ?? `transition_to_${toNodeId}`;
            const funcDescription =
                (edge["function_description"] as string) ??
                `Call this function when: ${edge["condition"] ?? ""}`;

            const parameters: Record<string, unknown> = {
                type: "object",
                properties: {},
                required: [],
            };

            const edgeParams = edge["parameters"] as Record<string, string> | undefined;
            if (edgeParams) {
                for (const [paramName, paramType] of Object.entries(edgeParams)) {
                    (parameters["properties"] as Record<string, unknown>)[paramName] = {
                        type: paramType,
                        description: `The ${paramName} provided by the user`,
                    };
                    (parameters["required"] as string[]).push(paramName);
                }
            }

            return {
                type: "function",
                function: { name: funcName, description: funcDescription, parameters },
            };
        });

        tools.push({
            type: "function",
            function: {
                name: "stay_on_current_node",
                description: "No transition matches. Need more info or clarification.",
                parameters: { type: "object", properties: {}, required: [] },
            },
        });

        if (nodeId) {
            if (this.transitionToolsCache.size >= this.transitionToolsCacheMaxSize) {
                const firstKey = this.transitionToolsCache.keys().next().value;
                if (firstKey) this.transitionToolsCache.delete(firstKey);
            }
            this.transitionToolsCache.set(nodeId, tools);
        }

        return tools;
    }

    private getEdgeByFunctionName(
        node: Record<string, unknown>,
        functionName: string
    ): Record<string, unknown> | null {
        const edges = (node["edges"] as Record<string, unknown>[]) ?? [];
        return (
            edges.find((edge) => {
                const expected =
                    (edge["function_name"] as string) ??
                    `transition_to_${edge["to_node_id"]}`;
                return expected === functionName;
            }) ?? null
        );
    }

    // ------------------------------------------------------------------
    // Node routing
    // ------------------------------------------------------------------

    async decideNextNodeWithFunctions(
        history: Record<string, unknown>[]
    ): Promise<RoutingDecision> {
        const start = performance.now();

        const currentNode = this.getNodeById(this.currentNodeId);
        if (!currentNode) {
            logger.error(`Current node '${this.currentNodeId}' not found`);
            return [null, null, 0, null, null];
        }

        const edges = (currentNode["edges"] as unknown[]) ?? [];
        if (!edges.length) {
            logger.debug(`Node '${this.currentNodeId}' has no edges, staying`);
            return [null, null, 0, null, null];
        }

        const tools = this.buildTransitionTools(currentNode);

        // Context section
        let contextSection = "";
        if (Object.keys(this.contextData).length) {
            const items = Object.entries(this.contextData)
                .filter(([k, v]) => v != null && typeof v !== "object" && k !== "detected_language")
                .map(([k, v]) => `${k}=${v}`);
            if (items.length) contextSection = `\nContext: ${items.join(", ")}`;
        }

        const defaultInstructions =
            "Call the transition function matching user intent, or stay_on_current_node if unclear.";
        let instructions = this.routingInstructions ?? defaultInstructions;

        if (this.contextData && instructions) {
            try {
                const sub: Record<string, unknown> = { ...this.contextData };
                const recipientData = this.contextData["recipient_data"];
                if (typeof recipientData === "object" && recipientData !== null) {
                    Object.assign(sub, recipientData);
                }
                instructions = instructions.replace(/\{(\w+)\}/g, (_, k) =>
                    k in sub ? String(sub[k]) : "NULL"
                );
            } catch (e) {
                logger.debug(`Variable substitution in routing_instructions failed: ${e}`);
            }
        }

        const nodeObjective = (currentNode["prompt"] as string) ?? "";
        const systemPrompt = `Routing Guidelines: \n ${instructions}\n Current Node: ${currentNode["id"]}${contextSection} \n Node Objective: ${nodeObjective}\n\n Node Conversation History:\n`;
        const messages: Record<string, unknown>[] = [{ role: "system", content: systemPrompt }];

        const nodeHistory =
            this.currentNodeEntryIndex < history.length
                ? history.slice(this.currentNodeEntryIndex)
                : history;

        const hasToolContext = nodeHistory.some(
            (m) => m["role"] === "assistant" && m["tool_calls"]
        );

        if (hasToolContext) {
            for (const msg of nodeHistory) {
                const role = msg["role"] as string;
                if (role === "assistant") {
                    if (msg["tool_calls"]) {
                        messages.push({ role: "assistant", content: null, tool_calls: msg["tool_calls"] });
                    } else if (msg["content"]) {
                        messages.push({ role: "assistant", content: msg["content"] });
                    }
                } else if (role === "tool") {
                    messages.push({ role: "tool", tool_call_id: msg["tool_call_id"] ?? "", content: msg["content"] ?? "" });
                } else if (role === "user" && msg["content"]) {
                    messages.push({ role: "user", content: msg["content"] });
                }
            }
        } else {
            for (const msg of nodeHistory) {
                const role = msg["role"] as string;
                if ((role === "user" || role === "assistant") && msg["content"]) {
                    messages.push({ role, content: msg["content"] });
                }
            }
        }

        if (messages.length === 1 && history.length) {
            const userMsg = history[history.length - 1]?.["content"];
            if (userMsg) messages.push({ role: "user", content: userMsg });
        }

        try {
            const routingKwargs: Record<string, unknown> = {
                model: this.routingModel,
                messages,
                tools,
                tool_choice: "required",
                parallel_tool_calls: false,
            };

            if (this.routingModel?.startsWith("gpt-5")) {
                routingKwargs["max_completion_tokens"] = this.routingMaxTokens ?? 150;
                routingKwargs["reasoning_effort"] =
                    this.routingReasoningEffort ?? process.env.GPT5_ROUTING_REASONING_EFFORT ?? "minimal";
            } else {
                routingKwargs["max_tokens"] = this.routingMaxTokens ?? 50;
                routingKwargs["temperature"] = 0.0;
            }

            const client = this.routingClient as {
                chat: { completions: { create: (args: unknown) => Promise<unknown> } };
            };
            const response = (await client.chat.completions.create(routingKwargs)) as Record<string, unknown>;
            const latencyMs = performance.now() - start;

            const choices = response["choices"] as Array<Record<string, unknown>>;
            const message = choices[0]?.["message"] as Record<string, unknown> | undefined;
            const toolCalls = message?.["tool_calls"] as Array<Record<string, unknown>> | undefined;

            if (toolCalls?.length) {
                const toolCall = toolCalls[0]!;
                const fn = toolCall["function"] as Record<string, unknown>;
                const functionName = fn["name"] as string;
                const functionArgs = fn["arguments"]
                    ? (JSON.parse(fn["arguments"] as string) as Record<string, unknown>)
                    : {};

                logger.info(`Routing decision: ${functionName} (latency: ${latencyMs.toFixed(1)}ms)`);

                if (functionName === "stay_on_current_node") {
                    return [null, null, latencyMs, messages, tools];
                }

                const edge = this.getEdgeByFunctionName(currentNode, functionName);
                if (edge) {
                    return [edge["to_node_id"] as string, functionArgs, latencyMs, messages, tools];
                }
                logger.warn(`Function ${functionName} not found in edges`);
                return [null, null, latencyMs, messages, tools];
            }

            logger.warn("No tool call in response");
            return [null, null, performance.now() - start, messages, tools];
        } catch (e) {
            const latencyMs = performance.now() - start;
            logger.error(`Routing error: ${e} (latency: ${latencyMs.toFixed(1)}ms)`);
            return [null, null, latencyMs, messages, tools];
        }
    }

    // ------------------------------------------------------------------
    // Node helpers
    // ------------------------------------------------------------------

    getNodeById(nodeId: string): Record<string, unknown> | null {
        const nodes = (this.config["nodes"] as Record<string, unknown>[]) ?? [];
        return nodes.find((n) => n["id"] === nodeId) ?? null;
    }

    private getPromptWithExample(node: Record<string, unknown>, detectedLang: string | undefined): string {
        const prompt = (node["prompt"] as string) ?? "";
        const examples = (node["examples"] as Record<string, string>) ?? {};
        if (!Object.keys(examples).length) return prompt;

        if (detectedLang && detectedLang in examples) {
            return `${prompt}\n\nLANGUAGE GUIDELINES\n\nPlease make sure to generate replies in the ${detectedLang} language only. You can refer to the example given below to generate a reply in the given language. Example response: "${examples[detectedLang]}"`;
        }

        const exampleLines = Object.entries(examples).map(([lang, text]) => `  ${lang.toUpperCase()}: "${text}"`);
        return `${prompt}\n\nExample responses:\n${exampleLines.join("\n")}`;
    }

    private getToolChoiceForNode(): unknown {
        if (!this.llm || !(this.llm as unknown as Record<string, unknown>)["triggerFunctionCall"]) {
            return null;
        }
        const currentNode = this.getNodeById(this.currentNodeId);
        if (!currentNode) return null;

        const fn = currentNode["function_call"] as string | undefined;
        if (fn) {
            logger.info(`Node '${this.currentNodeId}' forcing specific function: ${fn}`);
            return { type: "function", function: { name: fn } };
        }
        return null;
    }

    private async buildMessages(history: Record<string, unknown>[]): Promise<Record<string, unknown>[]> {
        const currentNode = this.getNodeById(this.currentNodeId);
        if (!currentNode) throw new Error("Current node not found.");

        const detectedLang = this.contextData["detected_language"] as string | undefined;
        let nodePrompt = this.getPromptWithExample(currentNode, detectedLang);

        if (this.contextData) {
            nodePrompt = updatePromptWithContext(nodePrompt, this.contextData);
        }

        let prompt = this.agentInformation
            ? `${updatePromptWithContext(this.agentInformation, this.contextData)}\n\n${nodePrompt}`
            : nodePrompt;

        // RAG augmentation
        const ragConfig = this.ragConfigs[this.currentNodeId];
        const collections = ragConfig?.["collections"] as string[] | undefined;
        if (ragConfig && collections?.length) {
            try {
                const client = RAGServiceClientSingleton.getClient(this.ragServerUrl);
                const latestMessage = (history[history.length - 1]?.["content"] as string) ?? "";
                const ragResponse = await client.queryForConversation(
                    latestMessage,
                    collections,
                    ragConfig["similarity_top_k"] as number ?? 10
                );
                if (ragResponse.contexts.length) {
                    const ragContext = client.formatContextForPrompt(ragResponse.contexts);
                    prompt = `${prompt}\n\nKnowledge base:\n${ragContext}\n\nUse this information naturally.`;
                }
            } catch (e) {
                logger.error(`RAG error for node ${this.currentNodeId}: ${e}`);
            }
        }

        const maxHistory = 50;
        const historySubset = history.slice(-maxHistory);
        const conversation = historySubset.filter((m) => m["role"] !== "system");
        return [{ role: "system", content: prompt }, ...conversation];
    }

    // ------------------------------------------------------------------
    // Generate
    // ------------------------------------------------------------------

    async *generate(
        message: Record<string, unknown>[],
        kwargs: Record<string, unknown> = {}
    ): AsyncGenerator<LLMStreamChunk | Record<string, unknown>> {
        const metaInfo = (kwargs["meta_info"] as Record<string, unknown>) ?? {};
        const synthesize = (kwargs["synthesize"] as boolean) ?? true;
        const startTime = nowMs();

        const detectedLanguage = metaInfo["detected_language"] as string | undefined;
        if (detectedLanguage) {
            this.contextData["detected_language"] = detectedLanguage;
        }

        try {
            const previousNode = this.currentNodeId;
            const [nextNodeId, extractedParams, routingLatencyMs, routingMessages, routingTools] =
                await this.decideNextNodeWithFunctions(message);

            if (nextNodeId) {
                logger.info(`Transitioning: ${this.currentNodeId} -> ${nextNodeId} (params: ${JSON.stringify(extractedParams)})`);
                this.currentNodeId = nextNodeId;
                this.currentNodeEntryIndex = message.length;
                if (extractedParams) Object.assign(this.contextData, extractedParams);
            }

            if (nextNodeId && this.nodeHistory[this.nodeHistory.length - 1] !== this.currentNodeId) {
                this.nodeHistory.push(this.currentNodeId);
            }

            yield {
                routing_info: {
                    previous_node: previousNode,
                    current_node: this.currentNodeId,
                    transitioned: nextNodeId !== null,
                    routing_model: this.routingModel,
                    routing_provider: this.routingProvider,
                    routing_latency_ms: Math.round(routingLatencyMs * 10) / 10,
                    extracted_params: extractedParams ?? {},
                    node_history: [...this.nodeHistory],
                    routing_messages: routingMessages,
                    routing_tools: routingTools,
                },
            };

            const messages = await this.buildMessages(message);
            yield { messages };

            const toolChoice = this.getToolChoiceForNode();
            yield* this.llm.generateStream(
                messages as never,
                synthesize,
                metaInfo as never,
                toolChoice as Record<string, unknown>
            ) as AsyncGenerator<LLMStreamChunk>;
        } catch (e) {
            logger.error(`Error in generate: ${e}`);
            const latencyData: LatencyData = {
                sequence_id: (metaInfo["sequence_id"] as number) ?? null,
                first_token_latency_ms: 0,
                total_stream_duration_ms: nowMs() - startTime,
                service_tier: null,
                llm_host: null,
            };
            yield {
                data: `An error occurred: ${e}`,
                end_of_stream: true,
                latency: latencyData,
                is_function_call: false,
            } as LLMStreamChunk;
        }
    }
}