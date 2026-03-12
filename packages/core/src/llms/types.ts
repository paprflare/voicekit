import { z } from "zod";

export const LatencyDataSchema = z.object({
    sequence_id: z.number().int().nullable().optional(),
    first_token_latency_ms: z.number().nullable().optional(),
    total_stream_duration_ms: z.number().nullable().optional(),
    service_tier: z.string().nullable().optional(),
    llm_host: z.string().nullable().optional(),
});

export const FunctionCallPayloadSchema = z.object({
    url: z.string().nullable().optional(),
    method: z.string().nullable().optional(),
    param: z.unknown().optional(),
    api_token: z.string().nullable().optional(),
    headers: z.record(z.any(), z.any()).nullable().optional(),
    model_args: z.record(z.any(), z.any()).default({}),
    meta_info: z.record(z.any(), z.any()).default({}),
    called_fun: z.string().default(""),
    model_response: z.array(z.record(z.any(), z.any())).default([]),
    tool_call_id: z.string().default(""),
    textual_response: z.string().nullable().optional(),
    resp: z.unknown().optional(),
}); // mirrors ConfigDict(extra='allow')

export const LLMStreamChunkSchema = z.object({
    data: z.unknown().optional(),
    end_of_stream: z.boolean().default(false),
    latency: LatencyDataSchema.nullable().optional(),
    is_function_call: z.boolean().default(false),
    function_name: z.string().nullable().optional(),
    function_message: z.union([z.string(), z.record(z.any(), z.any())]).nullable().optional(),
});

export type LatencyData = z.infer<typeof LatencyDataSchema>;
export type FunctionCallPayload = z.infer<typeof FunctionCallPayloadSchema>;
export type LLMStreamChunk = z.infer<typeof LLMStreamChunkSchema>;