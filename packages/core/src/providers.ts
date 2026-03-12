// Synthesizers
import { ElevenlabsSynthesizer } from "./synthesizers/elevenlabs.synthesizer";
import { OpenAISynthesizer } from "./synthesizers/openai.synthesizer";
import { DeepgramSynthesizer } from "./synthesizers/deepgram.synthesizer";
import { AzureSynthesizer } from "./synthesizers/asure.synthesizer";
import { CartesiaSynthesizer } from "./synthesizers/cartesia.synthesizer";
import { SarvamSynthesizer } from "./synthesizers/sarvam.synthesizer";

// Transcribers
import { DeepgramTranscriber } from "./transcribers/deepgram.transcriber";
import { AzureTranscriber } from "./transcribers/azure.transcriber";
import { SarvamTranscriber } from "./transcribers/sarvam.transcriber";
import { AssemblyAITranscriber } from "./transcribers/assemblyai.transcriber";
import { GoogleTranscriber } from "./transcribers/google.transcriber";
import { ElevenLabsTranscriber } from "./transcribers/elevenlabs.transcriber";


// LLMs
import { OpenAiLLM } from "./llms/openai.llm";
import { LiteLLM } from "./llms/litellm";


// Input handlers
import { DefaultInputHandler } from "./input-handlers/default";
import { TwilioInputHandler } from "./input-handlers/telephony-providers/twilio";
import { ExotelInputHandler } from "./input-handlers/telephony-providers/exotel";
import { PlivoInputHandler } from "./input-handlers/telephony-providers/plivo";
import { SipTrunkInputHandler } from "./input-handlers/telephony-providers/sip-tunk";

// Output handlers
import { DefaultOutputHandler } from "./output-handler/default";
import { TwilioOutputHandler } from "./output-handler/telephony-providers/twilio";
import { ExotelOutputHandler } from "./output-handler/telephony-providers/exotel";
import { PlivoOutputHandler } from "./output-handler/telephony-providers/plivo";
import { SipTrunkOutputHandler } from "./output-handler/telephony-providers/sip-tunk";
import { LLMProvider, SynthesizerProvider, TelephonyProvider, TranscriberProvider } from "./enums";




// ------------------------------------------------------------------
// Synthesizers
// ------------------------------------------------------------------

export const SUPPORTED_SYNTHESIZER_MODELS = {
    [SynthesizerProvider.elevenlabs]: ElevenlabsSynthesizer,
    [SynthesizerProvider.openai]: OpenAISynthesizer,
    [SynthesizerProvider.deepgram]: DeepgramSynthesizer,
    [SynthesizerProvider.azuretts]: AzureSynthesizer,
    [SynthesizerProvider.cartesia]: CartesiaSynthesizer,
    [SynthesizerProvider.sarvam]: SarvamSynthesizer,
};

// ------------------------------------------------------------------
// Transcribers
// ------------------------------------------------------------------

export const SUPPORTED_TRANSCRIBER_PROVIDERS = {
    [TranscriberProvider.deepgram]: DeepgramTranscriber,
    [TranscriberProvider.azure]: AzureTranscriber,
    [TranscriberProvider.sarvam]: SarvamTranscriber,
    [TranscriberProvider.assembly]: AssemblyAITranscriber,
    [TranscriberProvider.google]: GoogleTranscriber,
    [TranscriberProvider.elevenlabs]: ElevenLabsTranscriber,
};

// Backwards compatibility
export const SUPPORTED_TRANSCRIBER_MODELS = {
    deepgram: DeepgramTranscriber,
};

// ------------------------------------------------------------------
// LLMs
// ------------------------------------------------------------------

export const SUPPORTED_LLM_PROVIDERS = {
    [LLMProvider.openai]: OpenAiLLM,
    [LLMProvider.cohere]: LiteLLM,
    [LLMProvider.ollama]: LiteLLM,
    [LLMProvider.deepinfra]: LiteLLM,
    [LLMProvider.together]: LiteLLM,
    [LLMProvider.fireworks]: LiteLLM,
    [LLMProvider.perplexity]: LiteLLM,
    [LLMProvider.vllm]: LiteLLM,
    [LLMProvider.anyscale]: LiteLLM,
    [LLMProvider.custom]: OpenAiLLM,
    [LLMProvider.ola]: OpenAiLLM,
    [LLMProvider.groq]: LiteLLM,
    [LLMProvider.anthropic]: LiteLLM,
    [LLMProvider.deepseek]: LiteLLM,
    [LLMProvider.openrouter]: LiteLLM,
};

// ------------------------------------------------------------------
// Input handlers
// ------------------------------------------------------------------

export const SUPPORTED_INPUT_HANDLERS = {
    [TelephonyProvider.default]: DefaultInputHandler,
    [TelephonyProvider.twilio]: TwilioInputHandler,
    [TelephonyProvider.exotel]: ExotelInputHandler,
    [TelephonyProvider.plivo]: PlivoInputHandler,
    [TelephonyProvider["sip-trunk"]]: SipTrunkInputHandler,
};

export const SUPPORTED_INPUT_TELEPHONY_HANDLERS = {
    [TelephonyProvider.twilio]: TwilioInputHandler,
    [TelephonyProvider.exotel]: ExotelInputHandler,
    [TelephonyProvider.plivo]: PlivoInputHandler,
    [TelephonyProvider["sip-trunk"]]: SipTrunkInputHandler,
};

// ------------------------------------------------------------------
// Output handlers
// ------------------------------------------------------------------

export const SUPPORTED_OUTPUT_HANDLERS = {
    [TelephonyProvider.default]: DefaultOutputHandler,
    [TelephonyProvider.twilio]: TwilioOutputHandler,
    [TelephonyProvider.exotel]: ExotelOutputHandler,
    [TelephonyProvider.plivo]: PlivoOutputHandler,
    [TelephonyProvider["sip-trunk"]]: SipTrunkOutputHandler,
};

export const SUPPORTED_OUTPUT_TELEPHONY_HANDLERS = {
    [TelephonyProvider.twilio]: TwilioOutputHandler,
    [TelephonyProvider.exotel]: ExotelOutputHandler,
    [TelephonyProvider.plivo]: PlivoOutputHandler,
    [TelephonyProvider["sip-trunk"]]: SipTrunkOutputHandler,
};