import { z } from "zod";
import {
    TelephonyProviderSchema,
    SynthesizerProviderSchema,
    TranscriberProviderSchema,
    ReasoningEffortSchema,
    VerbositySchema,
} from "./enums";
import { ZodIssueCode } from "zod/v3";

// ============================================================
// Constants
// ============================================================

export const AGENT_WELCOME_MESSAGE =
    "This call is being recorded for quality assurance and training. Please speak now.";

// ============================================================
// Validation Helper
// ============================================================

export function validateAttribute(
    value: string,
    allowedValues: string[],
    valueType = "provider"
): string {
    if (!allowedValues.includes(value)) {
        throw new Error(
            `Invalid value for ${valueType}:'${value}' provided. Supported values: ${allowedValues}.`
        );
    }
    return value;
}

// ============================================================
// Synthesizer Provider Configs
// ============================================================

export const PollyConfigSchema = z.object({
    voice: z.string(),
    engine: z.string(),
    language: z.string(),
});

export const ElevenLabsConfigSchema = z.object({
    voice: z.string(),
    voice_id: z.string(),
    model: z.string(),
    temperature: z.number().optional().default(0.5),
    similarity_boost: z.number().optional().default(0.75),
    speed: z.number().optional().default(1.0),
    style: z.number().optional().default(0.0),
});

export const OpenAIConfigSchema = z.object({
    voice: z.string(),
    model: z.string(),
});

export const DeepgramConfigSchema = z.object({
    voice_id: z.string(),
    voice: z.string(),
    model: z.string(),
});

export const CartesiaConfigSchema = z.object({
    voice_id: z.string(),
    voice: z.string(),
    model: z.string(),
    language: z.string(),
    speed: z.number().optional().default(1.0),
});

export const RimeConfigSchema = z.object({
    voice_id: z.string(),
    language: z.string(),
    voice: z.string(),
    model: z.string(),
});

export const SmallestConfigSchema = z.object({
    voice_id: z.string(),
    language: z.string(),
    voice: z.string(),
    model: z.string(),
});

export const SarvamConfigSchema = z.object({
    voice_id: z.string(),
    language: z.string(),
    voice: z.string(),
    model: z.string(),
    speed: z.number().optional().default(1.0),
});

export const PixaConfigSchema = z.object({
    voice_id: z.string(),
    voice: z.string(),
    model: z.string(),
    language: z.string(),
    top_p: z.number().optional().default(0.95),
    repetition_penalty: z.number().optional().default(1.3),
});

export const AzureConfigSchema = z.object({
    voice: z.string(),
    model: z.string(),
    language: z.string(),
    speed: z.number().optional().default(1.0),
});

// ============================================================
// Transcriber
// ============================================================

export const TranscriberSchema = z.object({
    model: z.string().optional().default("nova-2"),
    language: z.string().nullable().optional(),
    stream: z.boolean().default(false),
    sampling_rate: z.number().int().optional().default(16000),
    encoding: z.string().optional().default("linear16"),
    endpointing: z.number().int().optional().default(500),
    keywords: z.string().nullable().optional(),
    task: z.string().optional().default("transcribe"),
    provider: z
        .string()
        .optional()
        .default("deepgram")
        .refine((v) => TranscriberProviderSchema.options.includes(v as never), {
            message: `Invalid transcriber provider. Supported: ${TranscriberProviderSchema.options}`,
        }),
});

// ============================================================
// Synthesizer
// ============================================================

const providerConfigMap = {
    elevenlabs: ElevenLabsConfigSchema,
    pixa: PixaConfigSchema,
    cartesia: CartesiaConfigSchema,
    polly: PollyConfigSchema,
    azuretts: AzureConfigSchema,
    deepgram: DeepgramConfigSchema,
    openai: OpenAIConfigSchema,
    smallest: SmallestConfigSchema,
    sarvam: SarvamConfigSchema,
    rime: RimeConfigSchema,
} as const;

export const SynthesizerSchema = z
    .object({
        provider: z
            .string()
            .refine((v) => SynthesizerProviderSchema.options.includes(v as never), {
                message: `Invalid synthesizer provider. Supported: ${SynthesizerProviderSchema.options}`,
            }),
        provider_config: z.record(z.any(), z.any()),
        stream: z.boolean().default(false),
        buffer_size: z.number().int().optional().default(40),
        audio_format: z.string().optional().default("pcm"),
        caching: z.boolean().optional().default(true),
    })
    .superRefine((data, ctx) => {
        const schema = providerConfigMap[data.provider as keyof typeof providerConfigMap];
        if (!schema) return;

        if (data.provider === "elevenlabs") {
            const cfg = data.provider_config as Record<string, unknown>;
            if (!cfg["voice"] || !cfg["voice_id"]) {
                ctx.addIssue({
                    code: z.ZodIssueCode.custom,
                    message: "ElevenLabs config requires 'voice' and 'voice_id'.",
                });
                return;
            }
        }

        const result = schema.safeParse(data.provider_config);
        if (!result.success) {
            result.error.issues.forEach((issue) => ctx.addIssue(issue as never));
        } else {
            data.provider_config = result.data;
        }
    });

// ============================================================
// IO Model
// ============================================================

export const IOModelSchema = z.object({
    provider: z
        .string()
        .refine((v) => TelephonyProviderSchema.options.includes(v as never), {
            message: `Invalid telephony provider. Supported: ${TelephonyProviderSchema.options}`,
        }),
    format: z.string().optional().default("wav"),
});

// ============================================================
// Vector Store
// ============================================================

export const MongoDBProviderConfigSchema = z.object({
    connection_string: z.string().nullable().optional(),
    db_name: z.string().nullable().optional(),
    collection_name: z.string().nullable().optional(),
    index_name: z.string().nullable().optional(),
    llm_model: z.string().optional().default("gpt-3.5-turbo"),
    embedding_model: z.string().optional().default("text-embedding-3-small"),
    embedding_dimensions: z.number().int().optional().default(256),
});

export const RerankerConfigSchema = z.object({
    enabled: z.boolean().default(false),
    model_type: z
        .string()
        .default("minilm-l6-v2")
        .refine((v) => ["bge-base", "bge-large", "bge-multilingual", "minilm-l6-v2"].includes(v), {
            message: "Invalid reranker model. Supported: bge-base, bge-large, bge-multilingual, minilm-l6-v2",
        }),
    candidate_count: z.number().int().min(1).max(100).default(20),
    final_count: z.number().int().min(1).max(50).default(5),
});

export const LanceDBProviderConfigSchema = z.object({
    vector_id: z.string(),
    similarity_top_k: z.number().int().optional().default(5),
    score_threshold: z.number().optional().default(0.1),
    reranker: RerankerConfigSchema.optional().default({
        candidate_count: 0,
        final_count: 0,
        enabled: false,
        model_type: "",
    }),
});

export const VectorStoreSchema = z.object({
    provider: z.string(),
    provider_config: z.union([LanceDBProviderConfigSchema, MongoDBProviderConfigSchema]),
});

// ============================================================
// LLM
// ============================================================

export const LlmSchema = z.object({
    model: z.string().optional().default("gpt-3.5-turbo"),
    max_tokens: z.number().int().optional().default(100),
    family: z.string().optional().default("openai"),
    temperature: z.number().optional().default(0.1),
    request_json: z.boolean().optional().default(false),
    stop: z.array(z.string()).nullable().optional(),
    top_k: z.number().int().optional().default(0),
    top_p: z.number().optional().default(0.9),
    min_p: z.number().optional().default(0.1),
    frequency_penalty: z.number().optional().default(0.0),
    presence_penalty: z.number().optional().default(0.0),
    provider: z.string().optional().default("openai"),
    base_url: z.string().nullable().optional(),
    reasoning_effort: ReasoningEffortSchema.nullable().optional(),
    verbosity: VerbositySchema.nullable().optional(),
    use_responses_api: z.boolean().optional().default(false),
});

export const SimpleLlmAgentSchema = LlmSchema.extend({
    agent_flow_type: z.string().optional().default("streaming"),
    extraction_details: z.string().nullable().optional(),
    summarization_details: z.string().nullable().optional(),
});

// ============================================================
// LLM Agent Graph (legacy)
// ============================================================

export const NodeSchema = z.object({
    id: z.string(),
    type: z.string(),
    llm: LlmSchema,
    exit_criteria: z.string(),
    exit_response: z.string().nullable().optional(),
    exit_prompt: z.string().nullable().optional(),
    is_root: z.boolean().optional().default(false),
});

export const EdgeSchema = z.object({
    start_node: z.string(),
    end_node: z.string(),
    condition: z.tuple([z.string(), z.string()]).nullable().optional(),
});

export const LlmAgentGraphSchema = z.object({
    nodes: z.array(NodeSchema),
    edges: z.array(EdgeSchema),
});

// ============================================================
// Graph Agent
// ============================================================

export const GraphEdgeSchema = z.object({
    to_node_id: z.string(),
    condition: z.string(),
    function_name: z.string().nullable().optional(),
    function_description: z.string().nullable().optional(),
    parameters: z.record(z.string(), z.any()).nullable().optional(),
});

export const GraphNodeRAGConfigSchema = z.object({
    vector_store: VectorStoreSchema,
    temperature: z.number().optional().default(0.7),
    model: z.string().optional().default("gpt-4"),
    max_tokens: z.number().int().optional().default(150),
});

export const GraphNodeSchema = z.object({
    id: z.string(),
    description: z.string().nullable().optional(),
    prompt: z.string(),
    examples: z.record(z.string(), z.any()).nullable().optional(),
    edges: z.array(GraphEdgeSchema).default([]),
    function_call: z.string().nullable().optional(),
    completion_check: z
        .function({
            input: [z.array(z.record(z.any(), z.any()))],
            output: z.boolean()
        })
        .nullable()
        .optional(),
    rag_config: GraphNodeRAGConfigSchema.nullable().optional(),
});

export const GraphAgentConfigSchema = LlmSchema.extend({
    agent_information: z.string(),
    nodes: z.array(GraphNodeSchema),
    current_node_id: z.string(),
    context_data: z.record(z.any(), z.any()).nullable().optional(),
    routing_model: z.string().nullable().optional(),
    routing_provider: z.string().nullable().optional(),
    routing_instructions: z.string().nullable().optional(),
    routing_reasoning_effort: ReasoningEffortSchema.nullable().optional(),
    routing_max_tokens: z.number().int().nullable().optional(),
});

// ============================================================
// Knowledge Agent
// ============================================================

export const KnowledgeAgentConfigSchema = LlmSchema.extend({
    agent_information: z.string().optional().default("Knowledge-based AI assistant"),
    prompt: z.string().nullable().optional(),
    rag_config: z.record(z.any(), z.any()).nullable().optional(),
    llm_provider: z.string().optional().default("openai"),
    context_data: z.record(z.any(), z.any()).nullable().optional(),
});

// ============================================================
// Multi-Agent
// ============================================================

export const AgentRouteConfigSchema = z.object({
    utterances: z.array(z.string()),
    threshold: z.number().optional().default(0.85),
});

export const KnowledgebaseAgentSchema = LlmSchema.extend({
    vector_store: VectorStoreSchema,
    provider: z.string().optional().default("openai"),
    model: z.string().optional().default("gpt-3.5-turbo"),
});

type LlmKey = z.infer<typeof LlmSchema>;

export const MultiAgentSchema = z.object({
    agent_map: z.record(z.string(), LlmSchema),
    agent_routing_config: z.record(z.any(), AgentRouteConfigSchema),
    default_agent: z.string(),
    embedding_model: z.string().optional().default("Snowflake/snowflake-arctic-embed-l"),
});

// ============================================================
// LLM Agent (discriminated by agent_type)
// ============================================================

const AGENT_TYPE_CONFIG_MAP = {
    knowledgebase_agent: KnowledgeAgentConfigSchema,
    graph_agent: GraphAgentConfigSchema,
    llm_agent_graph: LlmAgentGraphSchema,
    multiagent: MultiAgentSchema,
    simple_llm_agent: SimpleLlmAgentSchema,
} as const;

export type AgentType = keyof typeof AGENT_TYPE_CONFIG_MAP;

export const LlmAgentSchema = z
    .object({
        agent_flow_type: z.string(),
        agent_type: z.string(),
        llm_config: z.record(z.any(), z.any()),
    })
    .superRefine((data, ctx) => {
        const schema = AGENT_TYPE_CONFIG_MAP[data.agent_type as AgentType];
        if (!schema) {
            ctx.addIssue({
                code: ZodIssueCode.custom,
                message: `Unsupported agent_type: ${data.agent_type}. Supported: ${Object.keys(AGENT_TYPE_CONFIG_MAP)}`,
            });
            return;
        }
        const result = schema.safeParse(data.llm_config);
        if (!result.success) {
            result.error.issues.forEach((issue) => ctx.addIssue(issue.code));
        } else {
            data.llm_config = result.data;
        }
    });

// ============================================================
// Tools
// ============================================================

export const ToolFunctionSchema = z.object({
    name: z.string(),
    description: z.string(),
    parameters: z.record(z.any(), z.any()),
    strict: z.boolean().default(true),
});

export const ToolDescriptionSchema = z.object({
    type: z.string().default("function"),
    function: ToolFunctionSchema,
});

export const ToolDescriptionLegacySchema = z.object({
    name: z.string(),
    description: z.string(),
    parameters: z.record(z.any(), z.any()),
});

export const APIParamsSchema = z.object({
    url: z.string().nullable().optional(),
    method: z.string().optional().default("POST"),
    api_token: z.string().nullable().optional(),
    param: z.union([z.string(), z.record(z.any(), z.any())]).nullable().optional(),
    headers: z.union([z.string(), z.record(z.any(), z.any())]).nullable().optional(),
});

export const ToolModelSchema = z.object({
    tools: z
        .union([z.string(), z.array(z.union([ToolDescriptionSchema, ToolDescriptionLegacySchema]))])
        .nullable()
        .optional(),
    tools_params: z.record(z.any(), APIParamsSchema),
});

export const ToolsConfigSchema = z.object({
    llm_agent: z.union([LlmAgentSchema, SimpleLlmAgentSchema]).nullable().optional(),
    synthesizer: SynthesizerSchema.nullable().optional(),
    transcriber: TranscriberSchema.nullable().optional(),
    input: IOModelSchema.nullable().optional(),
    output: IOModelSchema.nullable().optional(),
    api_tools: ToolModelSchema.nullable().optional(),
});

// ============================================================
// Toolchain
// ============================================================

export const ToolsChainModelSchema = z.object({
    execution: z.enum(["parallel", "sequential"]),
    pipelines: z.array(z.array(z.string())),
});

// ============================================================
// Conversation Config
// ============================================================

export const ConversationConfigSchema = z.object({
    optimize_latency: z.boolean().optional().default(true),
    hangup_after_silence: z
        .number()
        .int()
        .nullable()
        .optional()
        .transform((v) => v ?? 10),
    incremental_delay: z.number().int().optional().default(900),
    number_of_words_for_interruption: z.number().int().optional().default(1),
    interruption_backoff_period: z.number().int().optional().default(100),
    hangup_after_LLMCall: z.boolean().optional().default(false),
    call_cancellation_prompt: z.string().nullable().optional(),
    backchanneling: z.boolean().optional().default(false),
    backchanneling_message_gap: z.number().int().optional().default(5),
    backchanneling_start_delay: z.number().int().optional().default(5),
    ambient_noise: z.boolean().optional().default(false),
    ambient_noise_track: z.string().optional().default("convention_hall"),
    call_terminate: z.number().int().optional().default(90),
    use_fillers: z.boolean().optional().default(false),
    trigger_user_online_message_after: z.number().int().optional().default(10),
    check_user_online_message: z
        .union([z.string(), z.record(z.string(), z.any())])
        .optional()
        .default("Hey, are you still there"),
    check_if_user_online: z.boolean().optional().default(true),
    generate_precise_transcript: z.boolean().optional().default(false),
    dtmf_enabled: z.boolean().optional().default(false),
    voicemail: z.boolean().optional().default(false),
    voicemail_detection_duration: z.number().optional().default(30.0),
    voicemail_check_interval: z.number().optional().default(7.0),
    voicemail_min_transcript_length: z.number().int().optional().default(7),
});

// ============================================================
// Task & Agent
// ============================================================

export const TaskSchema = z.object({
    tools_config: ToolsConfigSchema,
    toolchain: ToolsChainModelSchema,
    task_type: z.string().optional().default("conversation"),
    task_config: ConversationConfigSchema.default({
        optimize_latency: true,
        hangup_after_silence: 20,
        incremental_delay: 900,
        number_of_words_for_interruption: 3,
        interruption_backoff_period: 100,
        hangup_after_LLMCall: false,
        backchanneling: false,
        backchanneling_message_gap: 5,
        backchanneling_start_delay: 5,
        ambient_noise: false,
        ambient_noise_track: "convention_hall",
        call_terminate: 90,
        use_fillers: false,
        trigger_user_online_message_after: 10,
        check_user_online_message: "Hey, are you still there",
        check_if_user_online: true,
        generate_precise_transcript: false,
        dtmf_enabled: false,
        voicemail: false,
        voicemail_detection_duration: 30.0,
        voicemail_check_interval: 7.0,
        voicemail_min_transcript_length: 7,
    }),
});

export const AgentModelSchema = z.object({
    agent_name: z.string(),
    agent_type: z.string().default("other"),
    tasks: z.array(TaskSchema),
    agent_welcome_message: z.string().optional().default(AGENT_WELCOME_MESSAGE),
});


// ============================================================
// Inferred Types
// ============================================================

export type PollyConfig = z.infer<typeof PollyConfigSchema>;
export type ElevenLabsConfig = z.infer<typeof ElevenLabsConfigSchema>;
export type OpenAIConfig = z.infer<typeof OpenAIConfigSchema>;
export type DeepgramConfig = z.infer<typeof DeepgramConfigSchema>;
export type CartesiaConfig = z.infer<typeof CartesiaConfigSchema>;
export type RimeConfig = z.infer<typeof RimeConfigSchema>;
export type SmallestConfig = z.infer<typeof SmallestConfigSchema>;
export type SarvamConfig = z.infer<typeof SarvamConfigSchema>;
export type PixaConfig = z.infer<typeof PixaConfigSchema>;
export type AzureConfig = z.infer<typeof AzureConfigSchema>;
export type Transcriber = z.infer<typeof TranscriberSchema>;
export type Synthesizer = z.infer<typeof SynthesizerSchema>;
export type IOModel = z.infer<typeof IOModelSchema>;
export type RerankerConfig = z.infer<typeof RerankerConfigSchema>;
export type LanceDBProviderConfig = z.infer<typeof LanceDBProviderConfigSchema>;
export type MongoDBProviderConfig = z.infer<typeof MongoDBProviderConfigSchema>;
export type VectorStore = z.infer<typeof VectorStoreSchema>;
export type Llm = z.infer<typeof LlmSchema>;
export type SimpleLlmAgent = z.infer<typeof SimpleLlmAgentSchema>;
export type Node = z.infer<typeof NodeSchema>;
export type Edge = z.infer<typeof EdgeSchema>;
export type LlmAgentGraph = z.infer<typeof LlmAgentGraphSchema>;
export type GraphEdge = z.infer<typeof GraphEdgeSchema>;
export type GraphNodeRAGConfig = z.infer<typeof GraphNodeRAGConfigSchema>;
export type GraphNode = z.infer<typeof GraphNodeSchema>;
export type GraphAgentConfig = z.infer<typeof GraphAgentConfigSchema>;
export type KnowledgeAgentConfig = z.infer<typeof KnowledgeAgentConfigSchema>;
export type AgentRouteConfig = z.infer<typeof AgentRouteConfigSchema>;
export type MultiAgent = z.infer<typeof MultiAgentSchema>;
export type KnowledgebaseAgent = z.infer<typeof KnowledgebaseAgentSchema>;
export type LlmAgent = z.infer<typeof LlmAgentSchema>;
export type ToolFunction = z.infer<typeof ToolFunctionSchema>;
export type ToolDescription = z.infer<typeof ToolDescriptionSchema>;
export type ToolDescriptionLegacy = z.infer<typeof ToolDescriptionLegacySchema>;
export type APIParams = z.infer<typeof APIParamsSchema>;
export type ToolModel = z.infer<typeof ToolModelSchema>;
export type ToolsConfig = z.infer<typeof ToolsConfigSchema>;
export type ToolsChainModel = z.infer<typeof ToolsChainModelSchema>;
export type ConversationConfig = z.infer<typeof ConversationConfigSchema>;
export type Task = z.infer<typeof TaskSchema>;
export type AgentModel = z.infer<typeof AgentModelSchema>;