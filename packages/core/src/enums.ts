import { z } from "zod";

// ============================================================
// Chat
// ============================================================

export const ChatRoleSchema = z.enum(["system", "user", "assistant", "tool"]);
export type ChatRole = z.infer<typeof ChatRoleSchema>;

// ============================================================
// Telephony
// ============================================================

export const TelephonyProviderSchema = z.enum([
    "twilio",
    "exotel",
    "plivo",
    "sip-trunk",
    "default",
    "database",
]);
export type TelephonyProvider = z.infer<typeof TelephonyProviderSchema>;

export const TELEPHONY_ONLY_VALUES = [
    "twilio",
    "exotel",
    "plivo",
    "sip-trunk",
] as const satisfies TelephonyProvider[];

export const TelephonyOnlySchema = z.enum(TELEPHONY_ONLY_VALUES);
export const TelephonyProvider = TelephonyProviderSchema.enum;

// ============================================================
// Synthesizer (TTS)
// ============================================================

export const SynthesizerProviderSchema = z.enum([
    "polly",
    "elevenlabs",
    "openai",
    "deepgram",
    "azuretts",
    "cartesia",
    "sarvam",
]);
export type SynthesizerProvider = z.infer<typeof SynthesizerProviderSchema>;
export const SynthesizerProvider = SynthesizerProviderSchema.enum;

// ============================================================
// Transcriber (STT)
// ============================================================

export const TranscriberProviderSchema = z.enum([
    "deepgram",
    "azure",
    "sarvam",
    "assembly",
    "google",
    "elevenlabs",
]);
export type TranscriberProvider = z.infer<typeof TranscriberProviderSchema>;
export const TranscriberProvider = TranscriberProviderSchema.enum;

// ============================================================
// LLM
// ============================================================

export const LLMProviderSchema = z.enum([
    "openai",
    "cohere",
    "ollama",
    "deepinfra",
    "together",
    "fireworks",
    "azure-openai",
    "perplexity",
    "vllm",
    "anyscale",
    "custom",
    "ola",
    "groq",
    "anthropic",
    "deepseek",
    "openrouter",
    "azure",
]);
export type LLMProvider = z.infer<typeof LLMProviderSchema>;
export const LLMProvider = LLMProviderSchema.enum;

// ============================================================
// Reasoning & Verbosity
// ============================================================

// enums.ts

export const ReasoningEffortSchema = z.enum(["minimal", "low", "medium", "high"]);
export type ReasoningEffort = z.infer<typeof ReasoningEffortSchema>;
export const ReasoningEffort = ReasoningEffortSchema.enum;
// Usage: ReasoningEffort.LOW === "low"

export const VerbositySchema = z.enum(["low", "medium", "high"]);
export type Verbosity = z.infer<typeof VerbositySchema>;
export const Verbosity = VerbositySchema.enum;
// Usage: Verbosity.LOW === "low"

// ============================================================
// Responses API
// ============================================================

export const ResponseStreamEventSchema = z.enum([
    "response.created",
    "response.completed",
    "response.failed",
    "response.incomplete",
    "response.output_text.delta",
    "response.output_item.added",
    "response.function_call_arguments.delta",
]);
export type ResponseStreamEvent = z.infer<typeof ResponseStreamEventSchema>;

export const ResponseItemTypeSchema = z.enum([
    "message",
    "function_call",
    "function_call_output",
    "function",
]);
export type ResponseItemType = z.infer<typeof ResponseItemTypeSchema>;