// ============================================================
// Constants
// ============================================================

export const PREPROCESS_DIR = "agent_data";
export const PCM16_SCALE = 32768.0;

// ============================================================
// Model Prefixes
// ============================================================

export const GPT5_MODEL_PREFIX = "gpt-5";

// ============================================================
// Analytics
// ============================================================

export interface HighLevelAssistantAnalyticsData {
    extraction_details: Record<string, unknown>;
    cost_details: {
        average_transcriber_cost_per_conversation: number;
        average_llm_cost_per_conversation: number;
        average_synthesizer_cost_per_conversation: number;
    };
    historical_spread: {
        number_of_conversations_in_past_5_days: number[];
        cost_past_5_days: number[];
        average_duration_past_5_days: number[];
    };
    conversation_details: {
        total_conversations: number;
        finished_conversations: number;
        rejected_conversations: number;
    };
    execution_details: {
        total_conversations: number;
        total_cost: number;
        average_duration_of_conversation: number;
    };
    last_updated_at: string;
}

export const HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA: HighLevelAssistantAnalyticsData = {
    extraction_details: {},
    cost_details: {
        average_transcriber_cost_per_conversation: 0,
        average_llm_cost_per_conversation: 0,
        average_synthesizer_cost_per_conversation: 1.0,
    },
    historical_spread: {
        number_of_conversations_in_past_5_days: [],
        cost_past_5_days: [],
        average_duration_past_5_days: [],
    },
    conversation_details: {
        total_conversations: 0,
        finished_conversations: 0,
        rejected_conversations: 0,
    },
    execution_details: {
        total_conversations: 0,
        total_cost: 0,
        average_duration_of_conversation: 0,
    },
    last_updated_at: new Date().toISOString(),
};

// ============================================================
// Interruption / Filler Phrases
// ============================================================

export const ACCIDENTAL_INTERRUPTION_PHRASES: string[] = [
    "stop", "quit", "bye", "wait", "no", "wrong", "incorrect", "hold", "pause", "break",
    "cease", "halt", "silence", "enough", "excuse", "hold on", "hang on", "cut it",
    "that's enough", "shush", "listen", "excuse me", "hold up", "not now", "stop there",
    "stop speaking",
];

export const PRE_FUNCTION_CALL_MESSAGE: Record<string, string> = {
    en: "Just give me a moment, I'll be back with you.",
    ge: "Geben Sie mir einen Moment Zeit, ich bin gleich wieder bei Ihnen.",
};

export const FILLER_PHRASES: string[] = [
    "No worries.", "It's fine.", "I'm here.", "No rush.", "Take your time.",
    "Great!", "Awesome!", "Fantastic!", "Wonderful!", "Perfect!", "Excellent!",
    "I get it.", "Noted.", "Alright.", "I understand.", "Understood.", "Got it.",
    "Sure.", "Okay.", "Right.", "Absolutely.", "Sure thing.",
    "I see.", "Gotcha.", "Makes sense.",
];

export type FillerCategory =
    | "Unsure" | "Positive" | "Negative" | "Neutral" | "Explaining"
    | "Greeting" | "Farewell" | "Thanking" | "Apology" | "Clarification" | "Confirmation";

export const FILLER_DICT: Record<FillerCategory, string[]> = {
    Unsure: ["No worries.", "It's fine.", "I'm here.", "No rush.", "Take your time."],
    Positive: ["Great!", "Awesome!", "Fantastic!", "Wonderful!", "Perfect!", "Excellent!"],
    Negative: ["I get it.", "Noted.", "Alright.", "I understand.", "Understood.", "Got it."],
    Neutral: ["Sure.", "Okay.", "Right.", "Absolutely.", "Sure thing."],
    Explaining: ["I see.", "Gotcha.", "Makes sense."],
    Greeting: ["Hello!", "Hi there!", "Hi!", "Hey!"],
    Farewell: ["Goodbye!", "Thank you!", "Take care!", "Bye!"],
    Thanking: ["Welcome!", "No worries!"],
    Apology: ["I'm sorry.", "My apologies.", "I apologize.", "Sorry."],
    Clarification: ["Please clarify.", "Can you explain?", "More details?", "Can you elaborate?"],
    Confirmation: ["Got it.", "Okay.", "Understood."],
};

export const CHECKING_THE_DOCUMENTS_FILLER = "Umm, just a moment, getting details...";

export const TRANSFERING_CALL_FILLER: Record<string, string> = {
    en: "Sure, I'll transfer the call for you. Please wait a moment...",
    fr: "D'accord, je transfère l'appel. Un instant, s'il vous plaît.",
};

// ============================================================
// Defaults
// ============================================================

export const DEFAULT_USER_ONLINE_MESSAGE = "Hey, are you still there?";
export const DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION = 6;
export const DEFAULT_LANGUAGE_CODE = "en";
export const DEFAULT_TIMEZONE = "America/Los_Angeles";

// ============================================================
// Language Names
// ============================================================

export const LANGUAGE_NAMES: Record<string, string> = {
    en: "English", hi: "Hindi", bn: "Bengali",
    ta: "Tamil", te: "Telugu", mr: "Marathi",
    gu: "Gujarati", kn: "Kannada", ml: "Malayalam",
    pa: "Punjabi", fr: "French", es: "Spanish",
    pt: "Portuguese", de: "German", it: "Italian",
    nl: "Dutch", id: "Indonesian", ms: "Malay",
    th: "Thai", vi: "Vietnamese", od: "Odia",
};

// ============================================================
// LLM Defaults
// ============================================================

export interface LlmProviderConfig {
    model: string;
    provider: string;
}

export const LLM_DEFAULT_CONFIGS: Record<string, LlmProviderConfig> = {
    summarization: { model: "gpt-4.1-mini", provider: "openai" },
    extraction: { model: "gpt-4.1-mini", provider: "openai" },
};

// ============================================================
// Sarvam
// ============================================================

export const SARVAM_MODEL_SAMPLING_RATE_MAPPING: Record<string, number> = {
    "bulbul:v2": 22050,
    "bulbul:v3": 22050, // NOTE: Documentation claims 24000, but WAV header shows 22050
};