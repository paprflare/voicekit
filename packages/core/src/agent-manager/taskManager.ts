import { v4 as uuidv4 } from "uuid";
import { configureLogger } from "../helper/logger";
import {
    structureSystemPrompt,
    computeFunctionPreCallMessage,
    selectMessageByLanguage,
    getDateTimeFromTimezone,
    calculateAudioDuration,
    createWsDataPacket,
    getFileNamesInDirectory,
    getRawAudioBytes,
    isValidMd5,
    getRequiredInputTypes,
    formatMessages,
    getPromptResponses,
    resample,
    saveAudioFileToS3,
    updatePromptWithContext,
    getMd5Hash,
    cleanJsonString,
    wavBytesToPcm,
    convertToRequestLog,
    yieldChunksFromMemory,
    processTaskCancellation,
    pcmToUlaw,
} from "../helper/utils";
import { ConversationHistory, type Message } from "../helper/conversation.history";
import { MarkEventMetaData } from "../helper/markEvent.metadata";
import { ObservableVariable } from "../helper/observable.variable";
import { LanguageDetector } from "../helper/language.detector";
import { BaseManager } from "./base";
import { InterruptionManager } from "./interruption.manager";
import { triggerApi, computedApiResponse } from "../helper/function.calling.helper";
import {
    StreamingContextualAgent,
    ExtractionContextualAgent,
    SummarizationContextualAgent,
    WebhookAgent,
    GraphAgent,
    KnowledgeBaseAgent,
} from "../agent-types"
import {
    SUPPORTED_INPUT_HANDLERS,
    SUPPORTED_OUTPUT_HANDLERS,
    SUPPORTED_OUTPUT_TELEPHONY_HANDLERS,
    SUPPORTED_TRANSCRIBER_MODELS,
    SUPPORTED_TRANSCRIBER_PROVIDERS,
    SUPPORTED_SYNTHESIZER_MODELS,
    SUPPORTED_LLM_PROVIDERS,
} from "../providers";
import {
    ACCIDENTAL_INTERRUPTION_PHRASES,
    DEFAULT_USER_ONLINE_MESSAGE,
    DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION,
    DEFAULT_LANGUAGE_CODE,
    DEFAULT_TIMEZONE,
    LANGUAGE_NAMES,
    LLM_DEFAULT_CONFIGS,
} from "../constants";
import {
    CHECK_FOR_COMPLETION_PROMPT,
    VOICEMAIL_DETECTION_PROMPT,
    FILLER_PROMPT,
    DATE_PROMPT,
    EXTRACTION_PROMPT,
    SUMMARIZATION_PROMPT,
} from "../prompts";
import { TelephonyProvider } from "../enums";
import type { OpenAiLLM } from "../llms/openai.llm";
import type { BaseLLM } from "../llms/llm";
import type { GraphAgentConfig } from "../models";
import { ur } from "zod/v4/locales";

const logger = configureLogger("taskManager");

type ObserverFn<T> = (value: T) => void;

interface ConversationRecording {
    input: {
        data: Buffer
        started: number
    }
    output: Buffer[]
    metadata: {
        started: number
    }
}

// ---------------------------------------------------------------------------
// AsyncQueue — typed wrapper matching Python asyncio.Queue semantics
// ---------------------------------------------------------------------------
class AsyncQueue<T> {
    private items: T[] = [];
    private resolvers: ((item: T) => void)[] = [];

    async get(): Promise<T> {
        return new Promise((resolve) => {
            const item = this.items.shift();
            if (item !== undefined) resolve(item);
            else this.resolvers.push(resolve);
        });
    }

    putNowait(item: T): void {
        const resolver = this.resolvers.shift();
        if (resolver) resolver(item);
        else this.items.push(item);
    }

    async put(item: T): Promise<void> {
        this.putNowait(item);
    }

    empty(): boolean {
        return this.items.length === 0;
    }

    clear(): void {
        this.items = [];
        // don't clear resolvers — they're waiting callers
    }
}

// ---------------------------------------------------------------------------
// TaskManager
// ---------------------------------------------------------------------------
export class TaskManager extends BaseManager {
    // ── kwargs / config ──────────────────────────────────────────────────────
    kwargs: Record<string, unknown>;
    taskConfig: Record<string, unknown>;
    taskId: number;
    assistantName: string | undefined;

    // ── timing ───────────────────────────────────────────────────────────────
    conversationStartInitTs: number;
    startTime: number;

    // ── latencies ────────────────────────────────────────────────────────────
    llmLatencies: Record<string, unknown>;
    transcriberLatencies: Record<string, unknown>;
    synthesizerLatencies: Record<string, unknown>;
    ragLatencies: Record<string, unknown>;
    routingLatencies: Record<string, unknown>;
    streamSidTs: number | null = null;

    // ── locale ───────────────────────────────────────────────────────────────
    timezone: string;
    language: string;
    transferCallParams: unknown;

    // ── identity ─────────────────────────────────────────────────────────────
    assistantId: string | null;
    runId: string | undefined;
    callSid: string | null = null;
    streamSid: string | null = null;

    // ── websocket ────────────────────────────────────────────────────────────
    websocket: unknown;

    // ── context ──────────────────────────────────────────────────────────────
    contextData: Record<string, unknown> | null;
    inputParameters: Record<string, unknown> | null;

    // ── flags ────────────────────────────────────────────────────────────────
    turnBasedConversation: boolean;
    enforceStreaming: boolean;
    roomUrl: string | null;
    isWebBasedCall: boolean;
    yieldChunks: boolean;
    isLocal: boolean;
    textualChatAgent: boolean;
    stream: boolean;
    shouldRecord: boolean;
    conversationEnded: boolean;
    hangupTriggered: boolean;
    hangupTriggeredAt: number | null;
    hangupMessageQueued: boolean;
    endOfConversationInProgress: boolean;
    hangupMarkEventTimeout: number;
    nitro: boolean;
    defaultIo: boolean;
    outputHandlerSet: boolean;
    generatePreciseTranscript: boolean;
    endedByAssistant: boolean;
    shouldRespond: boolean;
    responseInPipeline: boolean;
    llmResponseGenerated: boolean;
    startedTransmittingAudio: boolean;
    askedIfUserIsStillThere: boolean;
    allowExtraSleep: boolean;
    checkIfUserOnline: boolean;
    shouldBackchannel: boolean;
    ambientNoise: boolean;
    voicemailDetectionEnabled: boolean;
    voicemailDetected: boolean;

    // ── timing state ─────────────────────────────────────────────────────────
    lastResponseTime: number;
    considerNextTranscriptAfter: number;
    lastTransmittedTimestamp: number;
    lastSpokenTimestamp: number;
    timeSinceLastSpokenHumanWord: number;

    // ── conversation state ───────────────────────────────────────────────────
    currentRequestId: string | null = null;
    previousRequestId: string | null = null;
    llmRejectedRequestIds: Set<string> = new Set();
    llmProcessedRequestIds: Set<string> = new Set();
    buffers: unknown[] = [];

    // ── prompts / history ────────────────────────────────────────────────────
    prompts: Record<string, unknown>;
    systemPrompt: { role: string; content: string };
    conversationHistory: ConversationHistory;
    labelFlow: unknown[];
    promptMap: Record<string, string> = {};

    // ── agent config ─────────────────────────────────────────────────────────
    llmConfig: Record<string, unknown> | null;
    llmAgentConfig: Record<string, unknown>;
    agentType: string | null;
    llmConfigMap: Record<string, Record<string, unknown>>;
    llmAgentMap: Record<string, unknown>;

    // ── tools ────────────────────────────────────────────────────────────────
    tools: Record<string, unknown>;

    // ── queues ───────────────────────────────────────────────────────────────
    audioQueue: AsyncQueue<unknown>;
    llmQueue: AsyncQueue<unknown>;
    synthesizerQueue: AsyncQueue<unknown>;
    transcriberOutputQueue: AsyncQueue<unknown>;
    dtmfQueue: AsyncQueue<unknown>;
    queues: Record<string, AsyncQueue<unknown>>;
    bufferedOutputQueue: AsyncQueue<unknown>;
    pipelines: unknown[][];

    // ── recording ────────────────────────────────────────────────────────────
    conversationRecording: ConversationRecording;

    // ── sampling ─────────────────────────────────────────────────────────────
    samplingRate: number;
    outputChunkSize: number;
    synthesizerProvider: string;
    synthesizerVoice: string;
    synthesizerCharacters: number;
    transcriberDuration: number;
    synthesizerCharactersCount: number;

    // ── feature config ───────────────────────────────────────────────────────
    minimumWaitDuration: number;
    incrementalDelay: number;
    hangConversationAfter: number;
    useFillers: boolean;
    useLlmToDetermineHangup: boolean;
    checkForCompletionPrompt: string | null;
    checkForCompletionLlm: string | undefined;
    callHangupMessageConfig: unknown;
    checkUserOnlineMessageConfig: unknown;
    triggerUserOnlineMessageAfter: number;
    numberOfWordsForInterruption: number;
    accidentalInterruptionPhrases: Set<string>;
    backchannelingStartDelay: number;
    backchannelingMessageGap: number;
    backchannelingAudios: string;
    filenames: string[];
    soundtrack: string;
    hangupDetail: string | null;
    firstMessagePassingTime: number | null = null;
    transcribedMessage: string;

    // ── voicemail ─────────────────────────────────────────────────────────────
    voicemailDetectionDuration: number;
    voicemailCheckInterval: number;
    voicemailMinTranscriptLength: number;
    voicemailDetectionPrompt: string;
    voicemailDetectionStartTime: number | null;
    voicemailLastCheckTime: number | null;
    voicemailLlm: string;

    // ── mark events ──────────────────────────────────────────────────────────
    markEventMetaData: MarkEventMetaData;
    observableVariables: Record<string, ObservableVariable<unknown>>;

    // ── interruption ─────────────────────────────────────────────────────────
    interruption_manager: InterruptionManager;

    // ── tasks (Promise handles) ───────────────────────────────────────────────
    llmTask: Promise<void> | null = null;
    executeFunctionCallTask: Promise<void> | null = null;
    synthesizerTasks: Promise<void>[] = [];
    synthesizerTask: Promise<void> | null = null;
    synthesizerMonitorTask: Promise<void> | null = null;
    dtmfTask: Promise<void> | null = null;
    outputTask: Promise<void> | null = null;
    hangupTask: Promise<void> | null = null;
    transcriptionTask: Promise<void> | null = null;
    transcriber_task: Promise<void> | null = null;
    firstMessageTask: Promise<void> | null = null;
    firstMessageTaskNew: Promise<void> | null = null;
    handleAccumulatedMessageTask: Promise<void> | null = null;
    backchannelingTask: Promise<void> | null = null;
    ambientNoiseTask: Promise<void> | null = null;
    voicemailCheckTask: Promise<void> | null = null;
    llmQueueTask: Promise<void> | null = null;

    // ── misc ─────────────────────────────────────────────────────────────────
    requestLogs: unknown[];
    conversationConfig: Record<string, unknown> | null;
    extractedData: unknown = null;
    summarizedData: unknown = null;
    webhookResponse: unknown = null;
    welcomeMessageAudio: string | null;
    preloadedWelcomeAudio: Buffer | null;
    cache: unknown;
    languageDetector: LanguageDetector;
    languageInjectionMode: string | undefined;
    languageInstructionTemplate: string | undefined;
    fuelPresetDirectory: string;

    // ── abort controllers ─────────────────────────────────────────────────────
    private _abortControllers: Map<string, AbortController> = new Map();

    constructor(opts: {
        agentName: string | undefined;
        taskId: number;
        task: Record<string, unknown>;
        websocket: unknown;
        inputParameters?: Record<string, unknown> | null;
        contextData?: Record<string, unknown> | null;
        assistantId?: string | null;
        turnBasedConversation?: boolean;
        cache?: unknown;
        inputQueue?: unknown;
        conversationHistory?: unknown;
        outputQueue?: unknown;
        yieldChunks?: boolean;
        [key: string]: unknown;
    }) {
        super();
        const {
            agentName, taskId, task, websocket,
            inputParameters, contextData, assistantId,
            turnBasedConversation, cache, inputQueue,
            conversationHistory, outputQueue, yieldChunks,
            ...rest
        } = opts;

        this.kwargs = { ...rest };
        this.kwargs["task_manager_instance"] = this;

        this.conversationStartInitTs = Date.now();
        this.startTime = Date.now() / 1000;

        this.llmLatencies = { connection_latency_ms: null, turn_latencies: [], other_latencies: [] };
        this.transcriberLatencies = { connection_latency_ms: null, turn_latencies: [] };
        this.synthesizerLatencies = { connection_latency_ms: null, turn_latencies: [] };
        this.ragLatencies = { turn_latencies: [] };
        this.routingLatencies = { turn_latencies: [] };

        this.taskConfig = task;
        this.timezone = DEFAULT_TIMEZONE;
        this.language = DEFAULT_LANGUAGE_CODE;
        this.transferCallParams = this.kwargs["transfer_call_params"] ?? null;

        const toolsConfig = (task["tools_config"] as Record<string, unknown>);
        if (toolsConfig["api_tools"] != null) {
            this.kwargs["api_tools"] = toolsConfig["api_tools"];
        }
        const llmAgent = toolsConfig["llm_agent"] as Record<string, unknown> | null;
        if (llmAgent?.["llm_config"] && (llmAgent["llm_config"] as Record<string, unknown>)["assistant_id"]) {
            this.kwargs["assistant_id"] = (llmAgent["llm_config"] as Record<string, unknown>)["assistant_id"];
        }

        logger.info(`doing task ${JSON.stringify(task)}`);

        this.taskId = taskId;
        this.assistantName = agentName;
        this.tools = {};
        this.websocket = websocket;
        this.contextData = contextData ?? null;
        this.turnBasedConversation = turnBasedConversation ?? false;
        this.enforceStreaming = (this.kwargs["enforce_streaming"] as boolean | undefined) ?? false;
        this.roomUrl = (this.kwargs["room_url"] as string | undefined) ?? null;
        this.isWebBasedCall = (this.kwargs["is_web_based_call"] as boolean | undefined) ?? false;
        this.yieldChunks = false;

        // Queues
        this.audioQueue = new AsyncQueue();
        this.llmQueue = new AsyncQueue();
        this.synthesizerQueue = new AsyncQueue();
        this.transcriberOutputQueue = new AsyncQueue();
        this.dtmfQueue = new AsyncQueue();
        this.queues = {
            dtmf: this.dtmfQueue,
            transcriber: this.audioQueue,
            llm: this.llmQueue,
            synthesizer: this.synthesizerQueue,
        };

        this.pipelines = (task["toolchain"] as Record<string, unknown>)["pipelines"] as unknown[][];
        this.textualChatAgent = false;

        // Identity
        this.assistantId = assistantId ?? null;
        this.runId = (this.kwargs["run_id"] as string | undefined);

        this.markEventMetaData = new MarkEventMetaData();
        this.samplingRate = 24000;
        this.conversationEnded = false;
        this.hangupTriggered = false;
        this.hangupTriggeredAt = null;
        this.hangupMessageQueued = false;
        this.endOfConversationInProgress = false;
        this.hangupMarkEventTimeout = 10;

        this.prompts = {};
        this.systemPrompt = { role: "system", content: "" };
        this.inputParameters = inputParameters ?? null;

        this.shouldRecord = false;
        this.conversationRecording = {
            input: { data: Buffer.alloc(0), started: Date.now() / 1000 },
            output: [],
            metadata: { started: 0 },
        };

        this.welcomeMessageAudio = (this.kwargs["welcome_message_audio"] as string | undefined) ?? null;
        delete this.kwargs["welcome_message_audio"];
        this.preloadedWelcomeAudio = this.welcomeMessageAudio
            ? Buffer.from(this.welcomeMessageAudio, "base64")
            : null;

        this.observableVariables = {};
        this.outputHandlerSet = false;

        // IO handlers
        if (taskId === 0) {
            if (this.isWebBasedCall) {
                (this.taskConfig["tools_config"] as Record<string, unknown>)["input"] = {
                    ...(toolsConfig["input"] as Record<string, unknown>),
                    provider: "default",
                };
                (this.taskConfig["tools_config"] as Record<string, unknown>)["output"] = {
                    ...(toolsConfig["output"] as Record<string, unknown>),
                    provider: "default",
                };
            }

            this.defaultIo =
                (toolsConfig["output"] as Record<string, unknown>)["provider"] === "default";

            const agentHangupObs = new ObservableVariable<unknown>(false);
            agentHangupObs.addObserver(this.agentHangupObserver.bind(this) as ObserverFn<any>);
            this.observableVariables["agent_hangup_observable"] = agentHangupObs;

            const finalChunkObs = new ObservableVariable<unknown>(false);
            finalChunkObs.addObserver(this.finalChunkPlayedObserver.bind(this) as ObserverFn<any>);
            this.observableVariables["final_chunk_played_observable"] = finalChunkObs;

            if (this.isWebBasedCall) {
                const initEventObs = new ObservableVariable<unknown>(null);
                initEventObs.addObserver(this.handleInitEvent.bind(this) as ObserverFn<any>);
                this.observableVariables["init_event_observable"] = initEventObs;
            }

            this.shouldRecord = !this.isWebBasedCall &&
                (toolsConfig["output"] as Record<string, unknown>)["provider"] === "default" &&
                this.enforceStreaming;

            this.setupInputHandlers(
                turnBasedConversation ?? false,
                inputQueue,
                this.shouldRecord
            );
        } else {
            this.defaultIo = false;
        }

        this.setupOutputHandlers(turnBasedConversation ?? false, outputQueue);

        // Kick off message task
        this.firstMessageTaskNew = this.messageTaskNew();

        this.conversationHistory = new ConversationHistory(conversationHistory as Message[]);
        this.labelFlow = [];

        // Task handles
        this.llmTask = null;
        this.executeFunctionCallTask = null;
        this.synthesizerTasks = [];
        this.synthesizerTask = null;
        this.synthesizerMonitorTask = null;
        this.dtmfTask = null;

        // Conversation state
        this.currentRequestId = null;
        this.previousRequestId = null;
        this.llmRejectedRequestIds = new Set();
        this.llmProcessedRequestIds = new Set();
        this.buffers = [];
        this.shouldRespond = false;
        this.lastResponseTime = Date.now() / 1000;
        this.considerNextTranscriptAfter = Date.now() / 1000;
        this.llmResponseGenerated = false;
        this.responseInPipeline = false;

        // Language detection
        this.languageDetector = new LanguageDetector(
            (task["task_config"] as Record<string, unknown>),
            this.runId
        );
        const taskConf = task["task_config"] as Record<string, unknown>;
        this.languageInjectionMode = taskConf["language_injection_mode"] as string | undefined;
        this.languageInstructionTemplate = taskConf["language_instruction_template"] as string | undefined;

        // Metering
        this.transcriberDuration = 0;
        this.synthesizerCharacters = 0;
        this.synthesizerCharactersCount = 0;
        this.endedByAssistant = false;

        // Defaults (overridden below for task 0)
        this.extractedData = null;
        this.summarizedData = null;
        this.stream =
            toolsConfig["synthesizer"] != null &&
            (toolsConfig["synthesizer"] as Record<string, unknown>)["stream"] === true &&
            (this.enforceStreaming || !this.turnBasedConversation);

        this.isLocal = false;
        this.llmConfig = null;
        this.agentType = null;
        this.llmConfigMap = {};
        this.llmAgentMap = {};
        this.llmAgentConfig = {};
        this.synthesizerProvider = "";
        this.synthesizerVoice = "";
        this.hangupDetail = null;
        this.transcribedMessage = "";
        this.requestLogs = [];
        this.conversationConfig = null;

        // Defaults for task-0 fields (initialised below)
        this.minimumWaitDuration = 400;
        this.incrementalDelay = 100;
        this.hangConversationAfter = 10;
        this.useFillers = false;
        this.useLlmToDetermineHangup = false;
        this.checkForCompletionPrompt = null;
        this.checkForCompletionLlm = process.env.CHECK_FOR_COMPLETION_LLM;
        this.callHangupMessageConfig = null;
        this.checkUserOnlineMessageConfig = DEFAULT_USER_ONLINE_MESSAGE;
        this.triggerUserOnlineMessageAfter = DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION;
        this.numberOfWordsForInterruption = 3;
        this.accidentalInterruptionPhrases = new Set(ACCIDENTAL_INTERRUPTION_PHRASES);
        this.backchannelingStartDelay = 5;
        this.backchannelingMessageGap = 2;
        this.backchannelingAudios = "";
        this.filenames = [];
        this.soundtrack = "";
        this.lastTransmittedTimestamp = 0;
        this.lastSpokenTimestamp = Date.now();
        this.timeSinceLastSpokenHumanWord = 0;
        this.askedIfUserIsStillThere = false;
        this.startedTransmittingAudio = false;
        this.allowExtraSleep = false;
        this.checkIfUserOnline = true;
        this.shouldBackchannel = false;
        this.ambientNoise = false;
        this.voicemailDetectionEnabled = false;
        this.voicemailDetected = false;
        this.voicemailDetectionDuration = 30.0;
        this.voicemailCheckInterval = 7.0;
        this.voicemailMinTranscriptLength = 7;
        this.voicemailDetectionPrompt = VOICEMAIL_DETECTION_PROMPT;
        this.voicemailDetectionStartTime = null;
        this.voicemailLastCheckTime = null;
        this.voicemailLlm = process.env.VOICEMAIL_DETECTION_LLM ?? "gpt-4.1-mini";
        this.outputChunkSize = 16384;
        this.nitro = true;
        this.generatePreciseTranscript = false;
        this.fuelPresetDirectory = "";

        // LLM config resolution
        if (this.isMultiagent()) {
            const agentMap = (llmAgent!["llm_config"] as Record<string, unknown>)["agent_map"] as Record<string, Record<string, unknown>>;
            for (const [agent, config] of Object.entries(agentMap)) {
                this.llmConfigMap[agent] = { ...config };
                this.llmConfigMap[agent]["buffer_size"] = (toolsConfig["synthesizer"] as Record<string, unknown>)["buffer_size"];
            }
        } else if (llmAgent != null) {
            if (this.isKnowledgebaseAgent() || this.isGraphAgent()) {
                const llmAgentConf = llmAgent;
                this.llmAgentConfig = llmAgentConf;
                const llmCfg = llmAgentConf["llm_config"] as Record<string, unknown>;
                this.llmConfig = {
                    model: llmCfg["model"],
                    max_tokens: llmCfg["max_tokens"],
                    provider: llmCfg["provider"],
                    buffer_size: (toolsConfig["synthesizer"] as Record<string, unknown>)?.["buffer_size"],
                    temperature: llmCfg["temperature"],
                };
            } else {
                const agentTypeProp = llmAgent["agent_type"] as string | undefined;
                if (!agentTypeProp) {
                    this.llmAgentConfig = llmAgent;
                } else {
                    this.llmAgentConfig = llmAgent["llm_config"] as Record<string, unknown>;
                }
                this.llmConfig = {
                    model: this.llmAgentConfig["model"],
                    max_tokens: this.llmAgentConfig["max_tokens"],
                    provider: this.llmAgentConfig["provider"],
                    temperature: this.llmAgentConfig["temperature"],
                };
            }

            if (this.llmAgentConfig["reasoning_effort"]) {
                this.llmConfig!["reasoning_effort"] = this.llmAgentConfig["reasoning_effort"];
            }
            if (this.llmAgentConfig["use_responses_api"]) {
                this.llmConfig!["use_responses_api"] = true;
            }
        }

        this.outputTask = null;
        this.bufferedOutputQueue = new AsyncQueue();
        this.cache = cache;
        this.interruption_manager = new InterruptionManager();
        this.hangupTask = null;

        // Task-0 specific setup
        if (taskId === 0) {
            const providerConfig = (toolsConfig["synthesizer"] as Record<string, unknown>)["provider_config"] as Record<string, unknown>;
            this.synthesizerVoice = providerConfig["voice"] as string;
            this.hangupDetail = null;
            this.handleAccumulatedMessageTask = null;
            this.hangupTask = null;
            this.transcriber_task = null;
            this.outputChunkSize = this.samplingRate === 24000 ? 16384 : 4096;
            this.nitro = true;
            this.conversationConfig = (task["task_config"] as Record<string, unknown>) ?? {};

            logger.info(`Conversation config ${JSON.stringify(this.conversationConfig)}`);
            this.generatePreciseTranscript = (this.conversationConfig["generate_precise_transcript"] as boolean | undefined) ?? false;

            const dtmfEnabled = (this.conversationConfig["dtmf_enabled"] as boolean | undefined) ?? false;
            if (dtmfEnabled) {
                (this.tools["input"] as Record<string, unknown>)["is_dtmf_active"] = true;
                this.dtmfTask = this.injectDigitsToConversation();
            }

            this.triggerUserOnlineMessageAfter = (this.conversationConfig["trigger_user_online_message_after"] as number | undefined) ?? DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION;
            this.checkIfUserOnline = (this.conversationConfig["check_if_user_online"] as boolean | undefined) ?? true;
            this.checkUserOnlineMessageConfig = (this.conversationConfig["check_user_online_message"] as unknown) ?? DEFAULT_USER_ONLINE_MESSAGE;

            if (this.checkUserOnlineMessageConfig && this.contextData) {
                if (typeof this.checkUserOnlineMessageConfig === "object" && !Array.isArray(this.checkUserOnlineMessageConfig)) {
                    const map = this.checkUserOnlineMessageConfig as Record<string, string>;
                    this.checkUserOnlineMessageConfig = Object.fromEntries(
                        Object.entries(map).map(([lang, msg]) => [lang, updatePromptWithContext(msg, this.contextData)])
                    );
                } else {
                    this.checkUserOnlineMessageConfig = updatePromptWithContext(this.checkUserOnlineMessageConfig as string, this.contextData);
                }
            }

            this.kwargs["process_interim_results"] =
                (this.conversationConfig["optimize_latency"] as boolean | undefined) === true ? "true" : "false";

            // Conversation timing config
            this.minimumWaitDuration = (toolsConfig["transcriber"] as Record<string, unknown>)["endpointing"] as number;
            this.lastSpokenTimestamp = Date.now();
            this.incrementalDelay = (this.conversationConfig["incremental_delay"] as number | undefined) ?? 100;
            this.hangConversationAfter = (this.conversationConfig["hangup_after_silence"] as number | undefined) ?? 10;
            this.lastTransmittedTimestamp = 0;
            this.useFillers = (this.conversationConfig["use_fillers"] as boolean | undefined) ?? false;
            this.useLlmToDetermineHangup = (this.conversationConfig["hangup_after_LLMCall"] as boolean | undefined) ?? false;
            this.checkForCompletionPrompt = null;

            if (this.useLlmToDetermineHangup) {
                this.checkForCompletionPrompt =
                    (this.conversationConfig["call_cancellation_prompt"] as string | undefined) ?? CHECK_FOR_COMPLETION_PROMPT;
                this.checkForCompletionPrompt += `\n                        Respond only in this JSON format:\n                            {{\n                              "hangup": "Yes" or "No"\n                            }}\n                    `;
            }

            this.callHangupMessageConfig = (this.conversationConfig["call_hangup_message"] as unknown) ?? null;
            if (this.callHangupMessageConfig && this.contextData && !this.isWebBasedCall) {
                if (typeof this.callHangupMessageConfig === "object") {
                    const map = this.callHangupMessageConfig as Record<string, string>;
                    this.callHangupMessageConfig = Object.fromEntries(
                        Object.entries(map).map(([lang, msg]) => [lang, updatePromptWithContext(msg, this.contextData)])
                    );
                } else {
                    this.callHangupMessageConfig = updatePromptWithContext(this.callHangupMessageConfig as string, this.contextData);
                }
            }

            this.checkForCompletionLlm = process.env.CHECK_FOR_COMPLETION_LLM;

            // Voicemail
            this.voicemailDetectionEnabled = (this.conversationConfig["voicemail"] as boolean | undefined) ?? false;
            this.voicemailLlm = process.env.VOICEMAIL_DETECTION_LLM ?? "gpt-4.1-mini";

            if (!("output" in this.tools) ||
                ((this.tools["output"] as Record<string, unknown>) &&
                    !(this.tools["output"] as { requiresCustomVoicemailDetection?: () => boolean }).requiresCustomVoicemailDetection?.())) {
                this.voicemailDetectionEnabled = false;
            }

            this.voicemailDetectionDuration = (this.conversationConfig["voicemail_detection_duration"] as number | undefined) ?? 30.0;
            this.voicemailCheckInterval = (this.conversationConfig["voicemail_check_interval"] as number | undefined) ?? 7.0;
            this.voicemailMinTranscriptLength = (this.conversationConfig["voicemail_min_transcript_length"] as number | undefined) ?? 7;
            this.voicemailDetectionPrompt = VOICEMAIL_DETECTION_PROMPT + `\n                    Respond only in this JSON format:\n                        {{\n                          "is_voicemail": "Yes" or "No"\n                        }}\n                `;
            this.voicemailDetected = false;
            this.voicemailDetectionStartTime = null;
            this.voicemailLastCheckTime = null;
            this.voicemailCheckTask = null;
            this.timeSinceLastSpokenHumanWord = 0;

            // Interruption config
            this.numberOfWordsForInterruption = (this.conversationConfig["number_of_words_for_interruption"] as number | undefined) ?? 3;
            this.askedIfUserIsStillThere = false;
            this.startedTransmittingAudio = false;
            this.accidentalInterruptionPhrases = new Set(ACCIDENTAL_INTERRUPTION_PHRASES);
            this.allowExtraSleep = false;

            this.interruption_manager = new InterruptionManager({
                numberOfWordsForInterruption: this.numberOfWordsForInterruption,
                accidentalInterruptionPhrases: ACCIDENTAL_INTERRUPTION_PHRASES,
                incrementalDelay: this.incrementalDelay,
                minimumWaitDuration: this.minimumWaitDuration,
            });

            // Backchanneling
            this.shouldBackchannel = (this.conversationConfig["backchanneling"] as boolean | undefined) ?? false;
            this.backchannelingTask = null;
            this.backchannelingStartDelay = (this.conversationConfig["backchanneling_start_delay"] as number | undefined) ?? 5;
            this.backchannelingMessageGap = (this.conversationConfig["backchanneling_message_gap"] as number | undefined) ?? 2;

            if (this.shouldBackchannel && !turnBasedConversation && taskId === 0) {
                logger.info("Should backchannel");
                this.backchannelingAudios = `${(this.kwargs["backchanneling_audio_location"] as string | undefined) ?? process.env.BACKCHANNELING_PRESETS_DIR}/${this.synthesizerVoice.toLowerCase()}`;
                try {
                    this.filenames = getFileNamesInDirectory(this.backchannelingAudios);
                    logger.info(`Backchanneling audio location ${this.backchannelingAudios}`);
                } catch (e) {
                    logger.error(`Something went wrong, putting shouldBackchannel to false: ${e}`);
                    this.shouldBackchannel = false;
                }
            }

            if ("agent_welcome_message" in this.kwargs) {
                logger.info(`Agent welcome message: ${this.kwargs["agent_welcome_message"]}`);
                this.firstMessageTask = null;
                this.transcribedMessage = "";
            }

            // Ambient noise
            this.ambientNoise = (this.conversationConfig["ambient_noise"] as boolean | undefined) ?? false;
            this.ambientNoiseTask = null;
            if (this.ambientNoise) {
                logger.info(`Ambient noise is True ${this.ambientNoise}`);
                this.soundtrack = `${(this.conversationConfig["ambient_noise_track"] as string | undefined) ?? "coffee-shop"}.wav`;
            }
        }

        // Setup transcriber + synthesizer
        this.setupTranscriber();
        this.setupSynthesizer(this.llmConfig);
        if (!this.turnBasedConversation && taskId === 0) {
            this.synthesizerMonitorTask = (this.tools["synthesizer"] as { monitorConnection: () => Promise<void> }).monitorConnection();
        }

        // Setup LLM + tasks
        if (this.llmConfig != null) {
            const llm = this.setupLlm(this.llmConfig, taskId);
            const agentTypeProp = this.llmAgentConfig["agent_type"] as string | undefined;
            this.setupTasks({ llm, agentType: agentTypeProp ?? "simple_llm_agent" });
        } else if (this.isMultiagent()) {
            const agentMap = (llmAgent!["llm_config"] as Record<string, unknown>)["agent_map"] as Record<string, Record<string, unknown>>;
            for (const agent of Object.keys(agentMap)) {
                if ("routes" in this.llmConfigMap[agent]!) {
                    delete this.llmConfigMap[agent]!["routes"];
                }
                const llm = this.setupLlm(this.llmConfigMap[agent]!);
                const agentTypeProp = this.llmConfigMap[agent]!["agent_type"] as string | undefined ?? "simple_llm_agent";
                logger.info(`Getting response for ${llm} and agent type ${agentTypeProp} and ${agent}`);
                const llmAgentObj = this.setupTasks({ llm, agentType: agentTypeProp });
                this.llmAgentMap[agent] = llmAgentObj;
            }
        } else if ((task["task_type"] as string) === "webhook") {
            const apiTools: any = toolsConfig["api_tools"];
            const webhookUrl = (apiTools["webhookURL"] as string | undefined) ??
                (apiTools["tools_params"])?.["webhook"]?.["url"] as string | undefined;
            logger.info(`Webhook URL ${webhookUrl}`);
            this.tools["webhook_agent"] = new WebhookAgent(webhookUrl!);
        }
    }

    // ── Properties ─────────────────────────────────────────────────────────

    get history(): unknown[] {
        return this.conversationHistory.messages;
    }

    set history(value: unknown[]) {
        (this.conversationHistory as unknown as { _messages: unknown[] })._messages = value;
    }

    get interimHistory(): unknown[] {
        return this.conversationHistory.interim;
    }

    set interimHistory(value: unknown[]) {
        (this.conversationHistory as unknown as { _interim: unknown[] })._interim = value;
    }

    get callHangupMessage(): string {
        const detectedLang = this.languageDetector?.dominantLanguage ?? null;
        return selectMessageByLanguage(this.callHangupMessageConfig as string | Record<string, string>, detectedLang);
    }

    // ── Private type helpers ────────────────────────────────────────────────

    private isMultiagent(): boolean {
        if ((this.taskConfig["task_type"] as string) === "webhook") return false;
        const llmAgent = (this.taskConfig["tools_config"] as Record<string, unknown>)["llm_agent"] as Record<string, unknown> | null;
        return (llmAgent?.["agent_type"] as string | undefined) === "multiagent";
    }

    private isKnowledgebaseAgent(): boolean {
        if ((this.taskConfig["task_type"] as string) === "webhook") return false;
        const llmAgent = (this.taskConfig["tools_config"] as Record<string, unknown>)["llm_agent"] as Record<string, unknown> | null;
        return (llmAgent?.["agent_type"] as string | undefined) === "knowledgebase_agent";
    }

    private isGraphAgent(): boolean {
        if ((this.taskConfig["task_type"] as string) === "webhook") return false;
        const llmAgent = (this.taskConfig["tools_config"] as Record<string, unknown>)["llm_agent"] as Record<string, unknown> | null;
        return (llmAgent?.["agent_type"] as string | undefined) === "graph_agent";
    }

    private invalidateResponseChain(): void {
        try {
            const llmAgent = this.tools["llm_agent"] as Record<string, unknown> | undefined;
            if (llmAgent && "llm" in llmAgent) {
                (llmAgent["llm"] as { invalidateResponseChain?: () => void }).invalidateResponseChain?.();
            }
        } catch (e) {
            logger.debug(`Failed to invalidate response chain: ${e}`);
        }
    }

    private injectLanguageInstruction(messages: Message[]): Message[] {
        const lang = this.languageDetector.dominantLanguage;
        if (!lang || !this.languageInjectionMode || !this.languageInstructionTemplate) return messages;

        try {
            const langName = LANGUAGE_NAMES[lang as keyof typeof LANGUAGE_NAMES] ?? lang;
            const instruction = this.languageInstructionTemplate.replace("{language}", langName) + "\n\n";

            if (this.languageInjectionMode === "system_only") {
                for (let i = 0; i < messages.length; i++) {
                    if (messages[i]!["role"] === "system") {
                        messages[i]!["content"] = instruction + (messages[i]!["content"] as string);
                        break;
                    }
                }
            } else if (this.languageInjectionMode === "per_turn") {
                for (let i = 0; i < messages.length; i++) {
                    if (messages[i]!["role"] === "user") {
                        messages[i]!["content"] = instruction + (messages[i]!["content"] as string);
                    }
                }
            }
        } catch (e) {
            logger.error(`Language injection error: ${e}`);
        }
        return messages;
    }

    // ── IO setup ────────────────────────────────────────────────────────────

    private setupOutputHandlers(turnBasedConversation: boolean, outputQueue: unknown): void {
        const toolsConfig = this.taskConfig["tools_config"] as Record<string, unknown>;
        const outputConfig = toolsConfig["output"] as Record<string, unknown> | null;
        const outputKwargs: Record<string, unknown> = { websocket: this.websocket };

        if (outputConfig == null) {
            logger.info("Not setting up any output handler as it is none");
            return;
        }

        const outputProvider = outputConfig["provider"] as string;
        if (!(outputProvider in SUPPORTED_OUTPUT_HANDLERS)) {
            throw new Error("Other output handlers not supported yet");
        }

        let outputHandlerClass: unknown;
        if (turnBasedConversation) {
            logger.info("Connected through dashboard, using default output handler");
            outputHandlerClass = SUPPORTED_OUTPUT_HANDLERS["default"];
        } else {
            outputHandlerClass = SUPPORTED_OUTPUT_HANDLERS[outputProvider as keyof typeof SUPPORTED_OUTPUT_HANDLERS];
            if (outputProvider in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS) {
                outputKwargs["mark_event_meta_data"] = this.markEventMetaData;
                logger.info("Making sure sampling rate for output handler is 8000");
                (toolsConfig["synthesizer"] as Record<string, unknown>)["provider_config"] = {
                    ...(toolsConfig["synthesizer"] as Record<string, unknown>)["provider_config"] as Record<string, unknown>,
                    sampling_rate: 8000,
                };

                if (outputProvider === TelephonyProvider["sip-trunk"]) {
                    (toolsConfig["synthesizer"] as Record<string, unknown>)["audio_format"] = "ulaw";
                    logger.info("Setting synthesizer audio format to ulaw for Asterisk sip-trunk");
                    const inputHandler = this.tools["input"];
                    outputKwargs["input_handler"] = inputHandler;
                    outputKwargs["asterisk_media_start"] = (this.contextData ?? {})["media_start_data"];
                    outputKwargs["agent_config"] = { tasks: [this.taskConfig] };
                    logger.info(`Passing input_handler to sip-trunk output handler: ${inputHandler != null}`);
                } else {
                    (toolsConfig["synthesizer"] as Record<string, unknown>)["audio_format"] = "pcm";
                }
            } else {
                (toolsConfig["synthesizer"] as Record<string, unknown>)["provider_config"] = {
                    ...(toolsConfig["synthesizer"] as Record<string, unknown>)["provider_config"] as Record<string, unknown>,
                    sampling_rate: 24000,
                };
                outputKwargs["queue"] = outputQueue;
            }

            this.samplingRate = ((toolsConfig["synthesizer"] as Record<string, unknown>)["provider_config"] as Record<string, unknown>)["sampling_rate"] as number;
        }

        if (outputProvider === "default") {
            outputKwargs["is_web_based_call"] = this.isWebBasedCall;
            outputKwargs["mark_event_meta_data"] = this.markEventMetaData;
        }

        const HandlerClass = outputHandlerClass as new (opts: Record<string, unknown>) => unknown;
        this.tools["output"] = new HandlerClass(outputKwargs);
        this.outputHandlerSet = true;
        logger.info("output handler set");
    }

    private setupInputHandlers(turnBasedConversation: boolean, inputQueue: unknown, shouldRecord: boolean): void {
        const toolsConfig = this.taskConfig["tools_config"] as Record<string, unknown>;
        const inputConfig = toolsConfig["input"] as Record<string, unknown>;
        const inputProvider = inputConfig["provider"] as string;

        if (!(inputProvider in SUPPORTED_INPUT_HANDLERS)) {
            throw new Error("Other input handlers not supported yet");
        }

        const inputKwargs: Record<string, unknown> = {
            queues: this.queues,
            websocket: this.websocket,
            input_types: getRequiredInputTypes(this.taskConfig),
            mark_event_meta_data: this.markEventMetaData,
            is_welcome_message_played:
                (toolsConfig["output"] as Record<string, unknown>)["provider"] === "default" && !this.isWebBasedCall,
        };

        if (shouldRecord) {
            inputKwargs["conversation_recording"] = this.conversationRecording;
        }

        let InputHandlerClass: unknown;
        if (this.turnBasedConversation) {
            inputKwargs["turn_based_conversation"] = true;
            InputHandlerClass = SUPPORTED_INPUT_HANDLERS["default"];
            inputKwargs["queue"] = inputQueue;
        } else {
            InputHandlerClass = SUPPORTED_INPUT_HANDLERS[inputProvider as keyof typeof SUPPORTED_INPUT_HANDLERS];
            if (inputProvider === "default") {
                inputKwargs["queue"] = inputQueue;
            }
            inputKwargs["observable_variables"] = this.observableVariables;

            if (inputProvider === TelephonyProvider["sip-trunk"] && this.contextData) {
                inputKwargs["ws_context_data"] = this.contextData;
                inputKwargs["agent_config"] = { tasks: [this.taskConfig] };
            }
        }

        const HandlerClass = InputHandlerClass as new (opts: Record<string, unknown>) => unknown;
        this.tools["input"] = new HandlerClass(inputKwargs);
    }

    private setupTranscriber(): void {
        try {
            const toolsConfig = this.taskConfig["tools_config"] as Record<string, unknown>;
            const transcriberConfig = toolsConfig["transcriber"] as Record<string, unknown> | null;
            if (transcriberConfig == null) return;

            this.language = (transcriberConfig["language"] as string | undefined) ?? DEFAULT_LANGUAGE_CODE;

            let provider: string;
            if (this.turnBasedConversation) provider = "playground";
            else if (this.isWebBasedCall) provider = "web_based_call";
            else provider = (toolsConfig["input"] as Record<string, unknown>)["provider"] as string;

            transcriberConfig["input_queue"] = this.audioQueue;
            transcriberConfig["output_queue"] = this.transcriberOutputQueue;

            if (provider === TelephonyProvider["sip-trunk"]) {
                transcriberConfig["encoding"] = "mulaw";
                transcriberConfig["sampling_rate"] = 8000;
                logger.info("Configured transcriber for Asterisk sip-trunk with mulaw encoding @ 8kHz");
            }

            if (transcriberConfig["model"] as keyof typeof SUPPORTED_TRANSCRIBER_MODELS in SUPPORTED_TRANSCRIBER_MODELS ||
                transcriberConfig["provider"] as keyof typeof SUPPORTED_TRANSCRIBER_PROVIDERS in SUPPORTED_TRANSCRIBER_PROVIDERS) {
                if (this.turnBasedConversation) {
                    transcriberConfig["stream"] = this.enforceStreaming;
                }

                let TranscriberClass: unknown;
                if ("provider" in transcriberConfig) {
                    TranscriberClass = SUPPORTED_TRANSCRIBER_PROVIDERS[transcriberConfig["provider"] as keyof typeof SUPPORTED_TRANSCRIBER_PROVIDERS];
                } else {
                    TranscriberClass = SUPPORTED_TRANSCRIBER_MODELS[transcriberConfig["model"] as keyof typeof SUPPORTED_TRANSCRIBER_MODELS];
                }

                const Cls = TranscriberClass as new (provider: string, opts: Record<string, unknown>) => unknown;
                this.tools["transcriber"] = new Cls(provider, { ...transcriberConfig, ...this.kwargs });
            }
        } catch (e) {
            logger.error(`Something went wrong with starting transcriber ${e}`);
        }
    }

    private setupSynthesizer(llmConfig: Record<string, unknown> | null): void {
        const toolsConfig = this.taskConfig["tools_config"] as Record<string, unknown>;

        if (this.isConversationTask()) {
            this.kwargs["use_turbo"] =
                (toolsConfig["transcriber"] as Record<string, unknown>)["language"] === DEFAULT_LANGUAGE_CODE;
        }

        const synthConfig = toolsConfig["synthesizer"] as Record<string, unknown> | null;
        if (synthConfig == null) return;

        const caching = ("caching" in synthConfig) ? (synthConfig["caching"] as boolean) : true;
        delete synthConfig["caching"];

        this.synthesizerProvider = synthConfig["provider"] as string;
        delete synthConfig["provider"];

        const SynthClass = SUPPORTED_SYNTHESIZER_MODELS[this.synthesizerProvider as keyof typeof SUPPORTED_SYNTHESIZER_MODELS];
        const providerConfig = synthConfig["provider_config"] as Record<string, unknown>;
        delete synthConfig["provider_config"];
        this.synthesizerVoice = providerConfig["voice"] as string;

        if (this.turnBasedConversation) {
            synthConfig["audio_format"] = "mp3";
            synthConfig["stream"] = this.enforceStreaming;
        }

        const synthesizerKwargs = { ...this.kwargs };
        if ((toolsConfig["output"] as Record<string, unknown>)["provider"] === TelephonyProvider["sip-trunk"]) {
            synthesizerKwargs["use_mulaw"] = true;
            logger.info("[SIP-TRUNK] Configuring synthesizer with use_mulaw=True");
        }

        const Cls = SynthClass as new (opts: Record<string, unknown>) => unknown;
        this.tools["synthesizer"] = new Cls({ ...synthConfig, ...providerConfig, ...synthesizerKwargs, caching });

        if (toolsConfig["llm_agent"] != null && llmConfig != null) {
            llmConfig["buffer_size"] = synthConfig["buffer_size"];
        }
    }

    private setupLlm(llmConfig: Record<string, unknown>, taskId = 0): unknown {
        const toolsConfig = this.taskConfig["tools_config"] as Record<string, unknown>;
        if (toolsConfig["llm_agent"] == null) return null;

        if (taskId > 0) {
            delete this.kwargs["llm_key"];
            delete this.kwargs["base_url"];
            delete this.kwargs["api_version"];

            if (this.isSummarizationTask() || this.isExtractionTask()) {
                llmConfig["model"] = LLM_DEFAULT_CONFIGS["summarization"]?.["model"];
                llmConfig["provider"] = LLM_DEFAULT_CONFIGS["summarization"]?.["provider"];
            }
        }

        const provider = llmConfig["provider"] as string;
        if (!(provider in SUPPORTED_LLM_PROVIDERS)) {
            throw new Error(`LLM ${provider} not supported`);
        }

        const LlmClass = SUPPORTED_LLM_PROVIDERS[provider as keyof typeof SUPPORTED_LLM_PROVIDERS];
        const {
            model,
            maxTokens,
            bufferSize,
            temperature
        } = llmConfig;

        return new LlmClass(
            model as string,
            maxTokens as number,
            bufferSize as number,
            temperature as number,
            this.language,
            { ...this.kwargs }
        );
    }

    private getAgentObject(llm: unknown, agentType: string): unknown {
        this.agentType = agentType;

        if (agentType === "simple_llm_agent") {
            return new StreamingContextualAgent(llm as OpenAiLLM);
        }

        if (agentType === "graph_agent") {
            logger.info("Setting up graph agent with rag-proxy-server support");
            const llmConfig = (this.taskConfig["tools_config"] as Record<string, unknown>)["llm_agent"] as Record<string, unknown>;
            const ragServerUrl = (this.kwargs["rag_server_url"] as string | undefined) ?? process.env.RAG_SERVER_URL ?? "http://localhost:8000";
            process.env.RAG_SERVER_URL = ragServerUrl;

            const injectedCfg: Record<string, unknown> = { ...((llmConfig["llm_config"] as Record<string, unknown>) ?? {}) };
            const passthroughKeys = ["llm_key", "base_url", "api_version", "api_tools", "reasoning_effort", "service_tier", "routing_reasoning_effort", "routing_max_tokens"];
            for (const key of passthroughKeys) {
                if (key in this.kwargs) injectedCfg[key] = this.kwargs[key];
            }
            if (this.contextData) injectedCfg["context_data"] = this.contextData;
            injectedCfg["buffer_size"] = (this.taskConfig["tools_config"] as Record<string, unknown>)["synthesizer"]
                ? ((this.taskConfig["tools_config"] as Record<string, unknown>)["synthesizer"] as Record<string, unknown>)["buffer_size"] : undefined;
            injectedCfg["language"] = this.language;

            return new GraphAgent(injectedCfg as GraphAgentConfig);
        }

        if (agentType === "knowledgebase_agent") {
            logger.info("Setting up knowledge agent with rag-proxy-server support");
            const llmConfig = (this.taskConfig["tools_config"] as Record<string, unknown>)["llm_agent"] as Record<string, unknown>;
            const ragServerUrl = (this.kwargs["rag_server_url"] as string | undefined) ?? process.env.RAG_SERVER_URL ?? "http://localhost:8000";
            process.env.RAG_SERVER_URL = ragServerUrl;

            const injectedCfg: Record<string, unknown> = { ...((llmConfig["llm_config"] as Record<string, unknown>) ?? {}) };
            const passthroughKeys = ["llm_key", "base_url", "api_version", "api_tools", "reasoning_effort", "service_tier"];
            for (const key of passthroughKeys) {
                if (key in this.kwargs) injectedCfg[key] = this.kwargs[key];
            }
            injectedCfg["buffer_size"] = (this.taskConfig["tools_config"] as Record<string, unknown>)["synthesizer"]
                ? ((this.taskConfig["tools_config"] as Record<string, unknown>)["synthesizer"] as Record<string, unknown>)["buffer_size"] : undefined;
            injectedCfg["language"] = this.language;

            return new KnowledgeBaseAgent(injectedCfg);
        }

        throw new Error(`${agentType} Agent type is not created yet`);
    }

    private setupTasks(opts: { llm?: unknown; agentType?: string }): unknown {
        const { llm, agentType } = opts;
        const taskType = this.taskConfig["task_type"] as string;

        if (taskType === "conversation" && !this.isMultiagent()) {
            this.tools["llm_agent"] = this.getAgentObject(llm, agentType ?? "simple_llm_agent");
        } else if (this.isMultiagent()) {
            return this.getAgentObject(llm, agentType ?? "simple_llm_agent");
        } else if (taskType === "extraction") {
            logger.info("Setting up extraction agent");
            this.tools["llm_agent"] = new ExtractionContextualAgent(llm as BaseLLM);
            this.extractedData = null;
        } else if (taskType === "summarization") {
            logger.info("Setting up summarization agent");
            this.tools["llm_agent"] = new SummarizationContextualAgent(llm as BaseLLM);
            this.summarizedData = null;
        }

        logger.info("prompt and config setup completed");
        return null;
    }

    // ── Initial message task ────────────────────────────────────────────────

    private async messageTaskNew(): Promise<void> {
        const tasks: Promise<void>[] = [];
        if (this.isConversationTask()) {
            tasks.push((this.tools["input"] as { handle: () => Promise<void> }).handle());
            if (!this.turnBasedConversation && !this.isWebBasedCall) {
                tasks.push(this.forcedFirstMessage());
            }
        }
        if (tasks.length) await Promise.all(tasks);
    }

    private async forcedFirstMessage(timeout = 10.0): Promise<void> {
        logger.info("Executing the first message task");
        try {
            const startTime = Date.now() / 1000;
            while (true) {
                const elapsed = Date.now() / 1000 - startTime;
                if (elapsed > timeout) {
                    await this.processEndOfConversation();
                    logger.warn("Timeout reached while waiting for stream_sid");
                    break;
                }

                const text = this.kwargs["agent_welcome_message"] as string | undefined ?? null;
                const metaInfo: Record<string, unknown> = {
                    io: (this.tools["output"] as { getProvider: () => string }).getProvider(),
                    message_category: "agent_welcome_message",
                    request_id: uuidv4(),
                    cached: true,
                    sequence_id: -1,
                    format: (this.taskConfig["tools_config"] as Record<string, unknown>)["output"]
                        ? ((this.taskConfig["tools_config"] as Record<string, unknown>)["output"] as Record<string, unknown>)["format"] : "pcm",
                    text,
                    end_of_llm_stream: true,
                };
                const wsDataPacket = createWsDataPacket({ data: text, metaInfo });
                const mi = wsDataPacket["meta_info"] as Record<string, unknown>;
                const pktText = wsDataPacket["data"] as string;
                mi["type"] = "audio";
                mi["synthesizer_start_time"] = Date.now() / 1000;

                let audioChunk: Buffer | null = this.preloadedWelcomeAudio;
                if (mi["text"] === "") audioChunk = null;

                const outputProvider = (this.tools["output"] as { getProvider: () => string }).getProvider();
                if (outputProvider === TelephonyProvider["sip-trunk"] && audioChunk) {
                    const originalSize = audioChunk.length;
                    audioChunk = pcmToUlaw(audioChunk);
                    logger.info(`[SIP-TRUNK] Converted welcome message PCM to ulaw: ${originalSize} bytes -> ${audioChunk.length} bytes`);
                    mi["format"] = "ulaw";
                } else {
                    mi["format"] = "pcm";
                }

                mi["is_first_chunk"] = true;
                mi["end_of_synthesizer_stream"] = true;
                mi["chunk_id"] = 1;
                mi["is_first_chunk_of_entire_response"] = true;
                mi["is_final_chunk_of_entire_response"] = true;
                const message = createWsDataPacket({ data: audioChunk, metaInfo: mi });

                const streamSid = (this.tools["input"] as { getStreamSid: () => string | null }).getStreamSid();
                if (streamSid != null && this.outputHandlerSet) {
                    this.streamSidTs = Date.now();
                    logger.info(`Got stream sid, sending first message ${streamSid}`);
                    this.streamSid = streamSid;
                    await (this.tools["output"] as { setStreamSid: (s: string) => Promise<void> }).setStreamSid(streamSid);
                    (this.tools["input"] as { updateIsAudioBeingPlayed: (v: boolean) => void }).updateIsAudioBeingPlayed(true);
                    convertToRequestLog({ message: pktText, metaInfo: mi, component: "synthesizer", direction: "response", model: this.synthesizerProvider, isCached: mi["is_cached"] as boolean, engine: (this.tools["synthesizer"] as { getEngine: () => string }).getEngine(), runId: this.runId });
                    await (this.tools["output"] as { handle: (msg: unknown) => Promise<void> }).handle(message);
                    try {
                        const data = (message)["data"];
                        if (data != null) {
                            const duration = calculateAudioDuration(
                                (data as Buffer).length,
                                this.samplingRate,
                                0,
                                0,
                                (message)["meta_info"]
                                    ? ((message)["meta_info"])["format"] as string | undefined : "pcm"
                            );
                            (this.conversationRecording["output"] as unknown[]).push({ data, start_time: Date.now() / 1000, duration });
                        }
                    } catch (e) {
                        logger.error(`Exception in forcedFirstMessage duration calculation: ${e}`);
                    }
                    break;
                } else {
                    logger.info(`Stream id still None (${streamSid}) or output handler not set (${this.outputHandlerSet}), waiting...`);
                    await new Promise((r) => setTimeout(r, 10));
                }
            }
        } catch (e) {
            logger.error(`Exception in forcedFirstMessage ${e}`);
        }
    }

    // ── Prompt loading ─────────────────────────────────────────────────────

    private getFinalPrompt(prompt: string, today: string, currentTime: string, currentTimezone: string): string {
        let enrichedPrompt = prompt;
        if (this.contextData != null) {
            enrichedPrompt = updatePromptWithContext(enrichedPrompt, this.contextData);
        }
        let notes = "### Note:\n";
        if (this.isConversationTask() && this.useFillers) {
            notes += `1.${FILLER_PROMPT}\n`;
        }
        return `${enrichedPrompt}\n${notes}\n${DATE_PROMPT(today, currentTime, currentTimezone)}`;
    }

    async loadPrompt(
        assistantName: string | undefined,
        taskId: number,
        opts: { local?: boolean;[key: string]: unknown }
    ): Promise<void> {
        if ((this.taskConfig["task_type"] as string) === "webhook") return;

        this.isLocal = opts.local ?? false;

        if (taskId === 0 && this.contextData?.["recipient_data"]) {
            const tz = (this.contextData["recipient_data"] as Record<string, unknown>)["timezone"] as string | undefined;
            if (tz) this.timezone = tz;
        }

        const [currentDate, currentTime] = getDateTimeFromTimezone(this.timezone);
        let promptResponses = opts["prompt_responses"] as Record<string, unknown> | null | undefined;
        if (!promptResponses) {
            promptResponses = await getPromptResponses(this.assistantId as string, this.isLocal) as Record<string, unknown> | null | undefined;
        }

        const currentTask = `task_${taskId + 1}`;

        if (this.isMultiagent()) {
            logger.info(`Getting ${currentTask} from prompt responses`);
            const prompts = (promptResponses as Record<string, unknown>)[currentTask] as Record<string, Record<string, unknown>>;
            this.promptMap = {};

            const llmAgent = (this.taskConfig["tools_config"] as Record<string, unknown>)["llm_agent"] as Record<string, unknown>;
            const agentMap = (llmAgent["llm_config"] as Record<string, unknown>)["agent_map"] as Record<string, Record<string, unknown>>;
            const defaultAgent = (llmAgent["llm_config"] as Record<string, unknown>)["default_agent"] as string;

            for (const agent of Object.keys(agentMap)) {
                let prompt = prompts[agent]!["system_prompt"] as string;
                prompt = this.prefillPrompts(this.taskConfig, prompt, this.taskConfig["task_type"] as string) as string;
                prompt = this.getFinalPrompt(prompt, currentDate, currentTime, this.timezone);
                if (agent === defaultAgent) {
                    this.systemPrompt = { role: "system", content: prompt };
                }
                this.promptMap[agent] = prompt;
            }
        } else {
            this.prompts = this.prefillPrompts(
                this.taskConfig,
                (promptResponses as Record<string, unknown>)[currentTask],
                this.taskConfig["task_type"] as string
            ) as Record<string, unknown>;
        }

        if ("system_prompt" in this.prompts) {
            let enrichedPrompt = this.prompts["system_prompt"] as string;

            if (this.contextData && (this.contextData["recipient_data"] as Record<string, unknown>)?.["call_sid"]) {
                this.callSid = (this.contextData["recipient_data"] as Record<string, unknown>)["call_sid"] as string;
            }

            enrichedPrompt = structureSystemPrompt({
                systemPrompt: enrichedPrompt,
                runId: this.runId as string,
                assistantId: this.assistantId as string,
                callSid: this.callSid,
                timezone: this.timezone,
                isWebBasedCall: this.isWebBasedCall,
                contextData: this.contextData
            });

            let notes = "";
            if (this.isConversationTask() && this.useFillers) {
                notes = "### Note:\n";
                notes += `1.${FILLER_PROMPT}\n`;
            }

            const finalPrompt = `\n## Agent Prompt:\n\n${enrichedPrompt}\n${notes}\n\n## Transcript:\n`;
            this.prompts["system_prompt"] = finalPrompt;
            this.systemPrompt = { role: "system", content: finalPrompt };
        } else {
            this.systemPrompt = { role: "system", content: "" };
        }

        let welcomeMsg = "";
        if (taskId === 0 && this.kwargs["agent_welcome_message"]) {
            welcomeMsg = this.kwargs["agent_welcome_message"] as string;
        }
        this.conversationHistory.setupSystemPrompt(this.systemPrompt, welcomeMsg);

        if (this.isKnowledgebaseAgent() && "llm_agent" in (this.taskConfig["tools_config"] as Record<string, unknown>)) {
            try {
                const llmAgentConf = (this.taskConfig["tools_config"] as Record<string, unknown>)["llm_agent"] as Record<string, unknown>;
                if ("llm_config" in llmAgentConf) {
                    (llmAgentConf["llm_config"] as Record<string, unknown>)["prompt"] = this.systemPrompt["content"];
                }
            } catch (e) {
                logger.error(`Failed to inject prompt into knowledge agent config: ${e}`);
            }
        }
    }

    private prefillPrompts(task: Record<string, unknown>, prompt: unknown, taskType: string): unknown {
        if (this.contextData?.["recipient_data"]) {
            const tz = (this.contextData["recipient_data"] as Record<string, unknown>)["timezone"] as string | undefined;
            if (tz) this.timezone = tz;
        }
        const [currentDate, currentTime] = getDateTimeFromTimezone(this.timezone);

        if (!prompt && (taskType === "extraction" || taskType === "summarization")) {
            if (taskType === "extraction") {
                const extractionJson =
                    (task as any)?.tools_config?.llm_agent?.llm_config?.extraction_json ?? "";

                const p = EXTRACTION_PROMPT(
                    currentDate,
                    currentTime,
                    this.timezone,
                    extractionJson
                );

                return { system_prompt: p };
            } else {
                return { system_prompt: SUMMARIZATION_PROMPT };
            }
        }
        return prompt;
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private processStopWords(textChunk: string, metaInfo: Record<string, unknown>): string {
        const lastFiveChars = textChunk.slice(-5).toLowerCase();
        const hasUser = lastFiveChars.includes("user");
        if (hasUser && metaInfo["end_of_llm_stream"]) {
            if (textChunk.slice(-5).toLowerCase() === "user:") return textChunk.slice(0, -5);
            if (textChunk.slice(-4).toLowerCase() === "user") return textChunk.slice(0, -4);
        }
        return textChunk;
    }

    updateTranscriptForInterruption(originalStream: string | null, heardText: string | null): string {
        if (originalStream == null) return heardText?.trim() ?? "";
        if (!heardText?.trim()) return "";

        const heard = heardText.trim();
        const index = originalStream.indexOf(heard);
        if (index !== -1) return originalStream.slice(0, index + heard.length);

        if (heard.length > 3 && originalStream.slice(0, 3) === heard.slice(0, 3)) {
            for (let i = heard.length; i > 0; i--) {
                const partial = heard.slice(0, i).trim();
                if (partial && originalStream.startsWith(partial)) return partial;
            }
        }
        return heard;
    }

    async syncHistory(markEventsData: [string, Record<string, unknown>][], interruptionProcessedAt: number): Promise<void> {
        try {
            const inputTool = this.tools["input"] as { responseHeardByUser: string; getCalculatedPlivoLatency: () => number; getCurrentMarkStartedTime: () => number };
            let responseHeard = inputTool.responseHeardByUser;
            logger.info(`sync_history: response_heard len=${responseHeard?.length ?? 0}`);

            if (!responseHeard) {
                const pendingMarks = markEventsData.map(([k, v]) => ({ mark_id: k, mark_data: v }));
                const pendingChunks: { text: string; duration: number; sent_ts: number }[] = [];

                for (const mark of pendingMarks) {
                    const markData = mark.mark_data;
                    const markType = markData["type"] as string;
                    const text = markData["text_synthesized"] as string;
                    if (["pre_mark_message", "ambient_noise", "backchanneling"].includes(markType) || !text) continue;
                    pendingChunks.push({
                        text,
                        duration: markData["duration"] as number ?? 0,
                        sent_ts: markData["sent_ts"] as number ?? 0,
                    });
                }

                if (pendingChunks.length) {
                    const firstSentTs = pendingChunks[0]!.sent_ts;
                    const plivoLatency = inputTool.getCalculatedPlivoLatency();
                    let actualPlayTime: number;

                    if (firstSentTs > 0) {
                        const timeSinceFirstSend = interruptionProcessedAt - firstSentTs;
                        actualPlayTime = Math.max(0, timeSinceFirstSend - plivoLatency);
                    } else {
                        const elapsed = interruptionProcessedAt - inputTool.getCurrentMarkStartedTime();
                        actualPlayTime = Math.max(0, elapsed - plivoLatency);
                    }

                    const playedText: string[] = [];
                    let cumDuration = 0;
                    for (const chunk of pendingChunks) {
                        if (cumDuration >= actualPlayTime) break;
                        if (cumDuration + chunk.duration <= actualPlayTime) {
                            playedText.push(chunk.text);
                        } else {
                            const remaining = actualPlayTime - cumDuration;
                            const proportion = chunk.duration > 0 ? remaining / chunk.duration : 0;
                            let partial = chunk.text.slice(0, Math.floor(chunk.text.length * proportion));
                            const lastSpace = partial.lastIndexOf(" ");
                            if (lastSpace > 0) partial = partial.slice(0, lastSpace);
                            if (partial) playedText.push(partial);
                        }
                        cumDuration += chunk.duration;
                    }

                    if (playedText.length) responseHeard = playedText.join("");
                } else {
                    return;
                }
            }

            this.conversationHistory.syncAfterInterruption(responseHeard, this.updateTranscriptForInterruption.bind(this));
            this.conversationHistory.syncInterimAfterInterruption(responseHeard, this.updateTranscriptForInterruption.bind(this));
            this.invalidateResponseChain();
        } catch (e) {
            logger.error(`sync_history failed: ${e}`);
        }
    }

    private async cleanupDownstreamTasks(): Promise<void> {
        const currentTs = Date.now() / 1000;
        logger.info("Cleaning up downstream task");
        const startTime = Date.now() / 1000;
        await (this.tools["output"] as { handleInterruption: () => Promise<void> }).handleInterruption();
        await (this.tools["synthesizer"] as { handleInterruption: () => Promise<void> }).handleInterruption();

        if (this.generatePreciseTranscript) {
            await this.syncHistory(
                Object.entries(this.markEventMetaData.fetchClearedMarkEventData()),
                currentTs
            );
            (this.tools["input"] as { resetResponseHeardByUser: () => void }).resetResponseHeardByUser();
        }

        this.interruption_manager.invalidatePendingResponses();
        this.responseInPipeline = false;
        await (this.tools["synthesizer"] as { flushSynthesizerStream: () => Promise<void> }).flushSynthesizerStream();

        if (this.outputTask != null) {
            logger.info("Cancelling output task");
            // No real cancel in TS - we use abort signals; just null the reference
            this.outputTask = null;
        }
        if (this.llmTask != null) {
            logger.info("Cancelling LLM Task");
            this.llmTask = null;
        }
        if (this.firstMessageTask != null) {
            logger.info("Cancelling first message task");
            this.firstMessageTask = null;
        }
        if (this.voicemailCheckTask != null) {
            logger.info("Cancelling voicemail check task");
            this.voicemailCheckTask = null;
        }

        if (!this.bufferedOutputQueue.empty()) {
            logger.info("Output queue was not empty, emptying it");
            this.bufferedOutputQueue = new AsyncQueue();
        }

        this.outputTask = this.processOutputLoop();
        this.startedTransmittingAudio = false;
        this.lastTransmittedTimestamp = Date.now() / 1000;
        logger.info(`Cleaning up downstream tasks. Time taken: ${Date.now() / 1000 - startTime}`);
    }

    private getUpdatedMetaInfo(metaInfo?: Record<string, unknown> | null): Record<string, unknown> {
        if (metaInfo == null) {
            metaInfo = (this.tools["transcriber"] as { getMetaInfo: () => Record<string, unknown> }).getMetaInfo();
            logger.info(`Metainfo ${JSON.stringify(metaInfo)}`);
        }
        const copy = { ...metaInfo };
        copy["sequence_id"] = this.interruption_manager.getNextSequenceId();
        copy["turn_id"] = this.interruption_manager.getTurnId();
        return copy;
    }

    private extractSequenceAndMeta(message: unknown): [number | null, Record<string, unknown> | null] {
        if (typeof message === "object" && message != null && "meta_info" in message) {
            this.setCallDetails(message as Record<string, unknown>);
            const metaInfo = (message as Record<string, unknown>)["meta_info"] as Record<string, unknown>;
            const sequence = (metaInfo["sequence"] as number) ?? 0;
            return [sequence, metaInfo];
        }
        return [null, null];
    }

    isExtractionTask(): boolean { return (this.taskConfig["task_type"] as string) === "extraction"; }
    isSummarizationTask(): boolean { return (this.taskConfig["task_type"] as string) === "summarization"; }
    isConversationTask(): boolean { return (this.taskConfig["task_type"] as string) === "conversation"; }

    private getNextStep(sequence: number, origin: string): string {
        try {
            const pipeline = this.pipelines[sequence] ?? [];
            for (let i = 0; i < pipeline.length - 1; i++) {
                if (pipeline[i] === origin) return pipeline[i + 1] as string;
            }
            return "output";
        } catch (e) {
            logger.error(`Error getting next step: ${e}`);
            return "output";
        }
    }

    private setCallDetails(message: Record<string, unknown>): void {
        if (this.callSid != null && this.streamSid != null) return;
        const mi = message["meta_info"] as Record<string, unknown> | undefined ?? {};
        if ("call_sid" in mi) this.callSid = mi["call_sid"] as string;
        if ("stream_sid" in mi) this.streamSid = mi["stream_sid"] as string;
    }

    // ── Followup task ────────────────────────────────────────────────────────

    private async processFollowupTask(message?: unknown): Promise<void> {
        if ((this.taskConfig["task_type"] as string) === "webhook") {
            const extractionDetails = (this.inputParameters ?? {})["extraction_details"] ?? {};
            logger.info(`DOING THE POST REQUEST TO WEBHOOK ${JSON.stringify(extractionDetails)}`);
            this.webhookResponse = await (this.tools["webhook_agent"] as { execute: (d: unknown) => Promise<unknown> }).execute(extractionDetails);
            logger.info(`Response from the server ${JSON.stringify(this.webhookResponse)}`);
        } else {
            const msgs = formatMessages((this.inputParameters ?? {})["messages"] as Message[], false, true);
            this.history.push({ role: "user", content: msgs });

            const startTime = Date.now();
            const jsonData = await (this.tools["llm_agent"] as { generate: (h: unknown[]) => Promise<Record<string, unknown>> }).generate(this.history as unknown[]);
            const latencyMs = Date.now() - startTime;

            if ((this.taskConfig["task_type"] as string) === "summarization") {
                this.summarizedData = jsonData["summary"];
                (this.llmLatencies["other_latencies"] as unknown[]).push({
                    type: "summarization", latency_ms: latencyMs,
                    model: LLM_DEFAULT_CONFIGS["summarization"]?.["model"],
                    provider: LLM_DEFAULT_CONFIGS["summarization"]?.["provider"],
                });
            } else {
                let data = cleanJsonString(jsonData);
                if (typeof data === "string") data = JSON.parse(data);
                this.extractedData = data;
                (this.llmLatencies["other_latencies"] as unknown[]).push({
                    type: "extraction", latency_ms: latencyMs,
                    model: LLM_DEFAULT_CONFIGS["extraction"]?.["model"],
                    provider: LLM_DEFAULT_CONFIGS["extraction"]?.["provider"],
                });
            }
        }
    }

    // ── Observers ───────────────────────────────────────────────────────────

    finalChunkPlayedObserver(_isFinalChunkPlayed: boolean): void {
        logger.info("Updating last_transmitted_timestamp");
        this.lastTransmittedTimestamp = Date.now() / 1000;
    }

    async agentHangupObserver(isAgentHangup: boolean): Promise<void> {
        logger.info(`agentHangupObserver triggered with is_agent_hangup = ${isAgentHangup}`);
        if (isAgentHangup) {
            (this.tools["output"] as { setHangupSent: () => void }).setHangupSent();
            await this.processEndOfConversation();
        }
    }

    async waitForCurrentMessage(): Promise<void> {
        const startTime = Date.now() / 1000;
        while (!this.conversationEnded) {
            const elapsed = Date.now() / 1000 - startTime;
            if (elapsed > this.hangupMarkEventTimeout) {
                const markEvents = this.markEventMetaData.getMarkEventMetaData;
                logger.warn(`waitForCurrentMessage timed out after ${this.hangupMarkEventTimeout}s with ${Object.keys(markEvents).length} remaining marks`);
                break;
            }

            const markEvents = this.markEventMetaData.getMarkEventMetaData;
            const markItemsList = Object.entries(markEvents).map(([k, v]) => ({ mark_id: k, mark_data: v }));
            logger.info(`current_list: ${JSON.stringify(markItemsList)}`);

            if (!markItemsList.length) break;

            const firstItem = markItemsList[0]!.mark_data as Record<string, unknown>;
            if (markItemsList.length === 1 && firstItem["type"] === "pre_mark_message") break;

            if (markItemsList.length === 2) {
                const secondItem = markItemsList[1]!.mark_data as Record<string, unknown>;
                if (firstItem["type"] === "agent_hangup" && firstItem["text_synthesized"] === "" && secondItem["type"] === "pre_mark_message") break;
            }

            if (firstItem["text_synthesized"] && firstItem["is_final_chunk"] === true) break;

            await new Promise((r) => setTimeout(r, 500));
        }
    }

    async injectDigitsToConversation(): Promise<void> {
        while (true) {
            try {
                const dtmfDigits = await this.queues["dtmf"]!.get() as string;
                logger.info(`DTMF collected ${dtmfDigits}`);
                const dtmfMessage = `dtmf_number: ${dtmfDigits}`;
                const baseMetaInfo: Record<string, unknown> = {
                    io: (this.tools["input"] as { ioProvider: string }).ioProvider,
                    type: "text",
                    sequence: 0,
                    origin: "dtmf",
                };
                const metaInfo = this.getUpdatedMetaInfo(baseMetaInfo);
                await this.handleTranscriberOutput("llm", dtmfMessage, metaInfo);
                logger.info(`DTMF LLM processing triggered with sequence_id=${metaInfo["sequence_id"]}`);
            } catch (e) {
                logger.info(`DTMF LLM processing triggered with exception ${e}`);
            }
        }
    }

    async processEndOfConversation(webCallTimeout = false): Promise<void> {
        if (this.endOfConversationInProgress || this.conversationEnded) {
            logger.info("processEndOfConversation: Already in progress or ended, skipping");
            return;
        }
        this.endOfConversationInProgress = true;
        logger.info("Got end of conversation. I'm stopping now");

        await this.waitForCurrentMessage();

        while (this.hangupTriggered && this.hangupMessageQueued) {
            try {
                if ((this.tools["output"] as { hangupSent: () => boolean }).hangupSent()) {
                    logger.info("final hangup chunk is now sent. Breaking now");
                    break;
                } else {
                    logger.info("final hangup chunk has not been sent yet");
                    await new Promise((r) => setTimeout(r, 500));
                }
            } catch (e) {
                logger.error(`Error while checking queue: ${e}`);
                break;
            }
        }

        if (this.hangupMessageQueued && !webCallTimeout) {
            this.history.push({ role: "assistant", content: this.callHangupMessage });
        }

        this.conversationEnded = true;
        this.endedByAssistant = true;

        if ("output" in this.tools && this.tools["output"] != null) {
            (this.tools["output"] as { close: () => void }).close();
        }

        await (this.tools["input"] as { stopHandler: () => Promise<void> }).stopHandler();
        logger.info("Stopped input handler");

        if ("transcriber" in this.tools && !this.turnBasedConversation) {
            logger.info("Stopping transcriber");
            await (this.tools["transcriber"] as { toggleConnection: () => Promise<void> }).toggleConnection();
            await new Promise((r) => setTimeout(r, 2_000));
        }

        if (this.voicemailCheckTask != null) {
            logger.info("Cancelling voicemail check task during conversation end");
            this.voicemailCheckTask = null;
        }
    }

    // ── LLM ─────────────────────────────────────────────────────────────────

    private async handleLlmOutput(
        nextStep: string,
        textChunk: string,
        shouldBypassSynth: boolean,
        metaInfo: Record<string, unknown>,
        isFiller = false,
        isFunctionCall = false
    ): Promise<void> {
        if (!("request_id" in metaInfo)) metaInfo["request_id"] = uuidv4();

        if (!this.stream && !isFiller) {
            const firstBufferLatency = Date.now() / 1000 - (metaInfo["llm_start_time"] as number);
            metaInfo["llm_first_buffer_generation_latency"] = firstBufferLatency;
        } else if (isFiller) {
            metaInfo["origin"] = "classifier";
            metaInfo["cached"] = true;
            metaInfo["local"] = true;
            metaInfo["message_category"] = "filler";
        }

        if (nextStep === "synthesizer" && !shouldBypassSynth) {
            const task = this.synthesize(createWsDataPacket({ data: textChunk, metaInfo }));
            this.synthesizerTasks.push(task);
        } else if (this.tools["output"] != null) {
            logger.info("Synthesizer not the next step, returning back");
            if (isFunctionCall) {
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: "<beginning_of_stream>", metaInfo })
                );
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: textChunk, metaInfo })
                );
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: "<end_of_stream>", metaInfo })
                );
            } else {
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: textChunk, metaInfo })
                );
            }
        }
    }

    private async doLlmGeneration(
        messages: Message[],
        metaInfo: Record<string, unknown>,
        nextStep: string,
        shouldBypassSynth = false,
        shouldTriggerFunctionCall = false
    ): Promise<void> {
        if (this.generatePreciseTranscript) {
            (this.tools["input"] as { resetResponseHeardByUser: () => void }).resetResponseHeardByUser();
        }

        let llmResponse = "";
        let functionTool = "";
        let functionToolMessage = "";
        const synthesize = !shouldBypassSynth;

        messages = this.injectLanguageInstruction(messages);
        const detectedLang = this.languageDetector.dominantLanguage;
        if (detectedLang) metaInfo["detected_language"] = detectedLang;

        const llmAgent = this.tools["llm_agent"] as {
            generate: (msgs: unknown[], opts: Record<string, unknown>) => AsyncGenerator<Record<string, unknown>>;
            checkForCompletion: (msgs: unknown[], prompt: string) => Promise<[Record<string, unknown>, Record<string, unknown>]>;
        };

        for await (const llmMessage of llmAgent.generate(messages, { synthesize, meta_info: metaInfo })) {
            // Custom message list pre-call
            if ("messages" in llmMessage) {
                convertToRequestLog({ message: formatMessages(llmMessage["messages"] as Message[], true, false), metaInfo, model: this.llmConfig?.["model"] as string, component: "llm", direction: "request", isCached: false, runId: this.runId });
                continue;
            }

            // Routing info (graph agent)
            if ("routing_info" in llmMessage) {
                const routingInfo = llmMessage["routing_info"] as Record<string, unknown>;
                // ... routing log handling (abbreviated for brevity)
                metaInfo["llm_metadata"] = metaInfo["llm_metadata"] ?? {};
                (metaInfo["llm_metadata"] as Record<string, unknown>)["graph_routing_info"] = routingInfo;
                if (routingInfo["routing_latency_ms"] != null) {
                    (this.routingLatencies["turn_latencies"] as unknown[]).push({
                        latency_ms: routingInfo["routing_latency_ms"],
                        routing_model: routingInfo["routing_model"],
                        previous_node: routingInfo["previous_node"],
                        current_node: routingInfo["current_node"],
                        transitioned: routingInfo["transitioned"] ?? false,
                        sequence_id: metaInfo["sequence_id"],
                    });
                }
                continue;
            }

            const data = (llmMessage["data"] as { data?: string } | string | undefined);
            const endOfLlmStream = llmMessage["end_of_stream"] as boolean | undefined;
            const latency = llmMessage["latency"] as Record<string, unknown> | undefined;
            const triggerFunctionCall = llmMessage["is_function_call"] as boolean | undefined;
            functionTool = (llmMessage["function_name"] as string | undefined) ?? "";
            functionToolMessage = (llmMessage["function_message"] as string | undefined) ?? "";

            if (triggerFunctionCall) {
                logger.info(`Triggering function call for ${JSON.stringify(data)}`);
                this.llmTask = this.executeFunctionCall({ nextStep, ...((data as Record<string, unknown>) ?? {}) });
                return;
            }

            if (latency) {
                const latencyDict = latency;
                const prevItem = (this.llmLatencies["turn_latencies"] as Record<string, unknown>[]).at(-1);
                if (prevItem?.["sequence_id"] === latencyDict["sequence_id"]) {
                    (this.llmLatencies["turn_latencies"] as Record<string, unknown>[]).splice(-1, 1, latencyDict);
                } else {
                    (this.llmLatencies["turn_latencies"] as unknown[]).push(latencyDict);
                }
            }

            const dataStr = typeof data === "string" ? data : (data as { data?: string })?.data ?? "";
            llmResponse += " " + dataStr;

            if (endOfLlmStream) metaInfo["end_of_llm_stream"] = true;

            if (this.stream) {
                const textChunk = this.processStopWords(dataStr, metaInfo);
                const lang = this.languageDetector.dominantLanguage ?? this.language;
                const fillerMessage = computeFunctionPreCallMessage(lang, functionTool, functionToolMessage);
                if (textChunk === fillerMessage) {
                    messages.push({ role: "assistant", content: fillerMessage });
                    this.conversationHistory.appendAssistant(fillerMessage);
                    this.conversationHistory.syncInterim(messages);
                }
                await this.handleLlmOutput(nextStep, textChunk, shouldBypassSynth, metaInfo);
            }
        }

        const lang = this.languageDetector.dominantLanguage ?? this.language;
        const fillerMessage = computeFunctionPreCallMessage(lang, functionTool, functionToolMessage);

        if (this.stream && llmResponse !== fillerMessage) {
            this.storeIntoHistory(metaInfo, messages, llmResponse, shouldTriggerFunctionCall);
        } else if (!this.stream) {
            llmResponse = llmResponse.trim();
            if (this.turnBasedConversation) this.conversationHistory.appendAssistant(llmResponse);
            await this.handleLlmOutput(nextStep, llmResponse, shouldBypassSynth, metaInfo, false, shouldTriggerFunctionCall);
            convertToRequestLog({ message: llmResponse, metaInfo, component: "llm", direction: "response", model: this.llmConfig?.["model"] as string, runId: this.runId });
        }

        if (metaInfo["rag_latency"]) {
            const ragLatency = metaInfo["rag_latency"] as Record<string, unknown>;
            const existingSeqIds = (this.ragLatencies["turn_latencies"] as Record<string, unknown>[]).map((t) => t["sequence_id"]);
            if (!existingSeqIds.includes(ragLatency["sequence_id"])) {
                (this.ragLatencies["turn_latencies"] as unknown[]).push(ragLatency);
            }
        }
    }

    private storeIntoHistory(
        metaInfo: Record<string, unknown>,
        messages: Message[],
        llmResponse: string,
        shouldTriggerFunctionCall = false
    ): void {
        this.llmResponseGenerated = true;
        convertToRequestLog({ message: llmResponse, metaInfo, component: "llm", direction: "response", model: this.llmConfig?.["model"] as string, runId: this.runId });
        if (shouldTriggerFunctionCall) {
            this.conversationHistory.appendAssistant(llmResponse);
        } else {
            messages.push({ role: "assistant", content: llmResponse });
            this.conversationHistory.appendAssistant(llmResponse);
            this.conversationHistory.syncInterim(messages);
        }
    }

    private async executeFunctionCall(opts: Record<string, unknown>): Promise<void> {
        const { url, method, param, apiToken, headers, modelArgs, metaInfo, nextStep, calledFun, ...resp } = opts as {
            url: string; method: string; param: unknown; apiToken: string;
            headers: Record<string, string>; modelArgs: unknown;
            metaInfo: Record<string, unknown>; nextStep: string; calledFun: string;
            [key: string]: unknown;
        };

        this.checkIfUserOnline = false;

        if (calledFun?.startsWith("transfer_call")) {
            await new Promise((r) => setTimeout(r, 2_000));
            let fromNumber: string | undefined;
            try { fromNumber = (this.contextData?.["recipient_data"] as Record<string, unknown>)?.["from_number"] as string; } catch { /* ignore */ }

            const payload: Record<string, unknown> = {
                call_sid: null,
                provider: (this.tools["input"] as { ioProvider: string }).ioProvider,
                stream_sid: this.streamSid,
                from_number: fromNumber,
                execution_id: this.runId,
                ...((this.transferCallParams as Record<string, unknown>) ?? {}),
            };

            const finalUrl = url ?? process.env.CALL_TRANSFER_WEBHOOK_URL;
            if ((this.tools["input"] as { ioProvider: string }).ioProvider !== "default") {
                payload["call_sid"] = (this.tools["input"] as { getCallSid: () => string }).getCallSid();
            }

            if (param != null) Object.assign(payload, resp);

            if ((this.tools["input"] as { ioProvider: string }).ioProvider === "default") {
                const mockResponse = `This is a mocked response demonstrating a successful transfer of call`;
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: "<beginning_of_stream>", metaInfo })
                );
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: mockResponse, metaInfo })
                );
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: "<end_of_stream>", metaInfo })
                );
                return;
            }

            while ((this.tools["input"] as { isAudioBeingPlayedToUser: () => boolean }).isAudioBeingPlayedToUser()) {
                await new Promise((r) => setTimeout(r, 1_000));
            }

            const res = await fetch(finalUrl!, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
            const responseText = await res.text();
            logger.info(`Response from server after call transfer: ${responseText}`);
            return;
        }

        const response = await triggerApi(url, method.toLowerCase(), param as string | Record<string, unknown> | null, apiToken, headers, metaInfo, this.runId as string, resp);
        const functionResponse = String(response);
        const [getResKeys, getResValues] = await computedApiResponse(functionResponse);

        if (this.isGraphAgent()) {
            try {
                const responseData = typeof functionResponse === "string" ? JSON.parse(functionResponse) : functionResponse;
                if (typeof responseData === "object" && responseData != null) {
                    this.contextData = this.contextData ?? {};
                    Object.assign(this.contextData, responseData);
                    const llmAgentInst = this.tools["llm_agent"] as { contextData?: Record<string, unknown> };
                    if (llmAgentInst.contextData) Object.assign(llmAgentInst.contextData, responseData);
                }
            } catch { /* ignore */ }
        }

        let setResponsePrompt: unknown = functionResponse;
        if (calledFun?.startsWith("check_availability_of_slots") && (!getResValues || (getResValues.length === 1 && (getResValues[0] as string)!.length === 0))) {
            setResponsePrompt = [];
        } else if (calledFun?.startsWith("book_appointment") && getResKeys?.includes("id")) {
            if (getResValues?.[0] === "no_available_users_found_error") {
                setResponsePrompt = "Sorry, the host isn't available at this time. Are you available at any other time?";
            } else {
                setResponsePrompt = [];
            }
        }

        const textualResponse = (resp as Record<string, unknown>)["textual_response"] as string | undefined;
        this.conversationHistory.appendAssistant(textualResponse as string, resp["model_response"] as Record<string, unknown>[]);
        this.conversationHistory.appendToolResult((resp["tool_call_id"] as string) ?? "", functionResponse);

        convertToRequestLog({
            message: functionResponse,
            metaInfo: metaInfo,
            model: this.llmConfig?.["model"] as string,
            component: "function_call",
            direction: "response",
            engine: this.llmConfig?.["engine"] as string,
            isCached: false,
            runId: this.runId
        });

        const messages = this.conversationHistory.getCopy();
        convertToRequestLog({ message: formatMessages(messages as Message[], true), metaInfo, model: this.llmConfig?.["model"] as string, component: "llm", direction: "request", isCached: false, runId: this.runId });

        this.checkIfUserOnline = (this.conversationConfig?.["check_if_user_online"] as boolean | undefined) ?? true;

        if (!calledFun?.startsWith("transfer_call")) {
            const shouldBypassSynth = (metaInfo["bypass_synth"] as boolean | undefined) ?? false;
            await this.doLlmGeneration(messages as Message[], metaInfo, nextStep, shouldBypassSynth, true);
        }

        this.executeFunctionCallTask = null;
    }

    private async processConversationTask(
        message: Record<string, unknown>,
        sequence: number,
        metaInfo: Record<string, unknown>
    ): Promise<void> {
        const shouldBypassSynth = metaInfo["bypass_synth"] === true;
        const nextStep = this.getNextStep(sequence, "llm");
        metaInfo["llm_start_time"] = Date.now() / 1000;

        if (this.turnBasedConversation) {
            this.history.push({ role: "user", content: (message["data"] as string) });
        }
        const messages = structuredClone(this.history) as Message[];

        if (!this.isKnowledgebaseAgent() && !this.isGraphAgent()) {
            convertToRequestLog({ message: formatMessages(messages as Message[], true), metaInfo, component: "llm", direction: "request", model: this.llmConfig?.["model"] as string, runId: this.runId });
        }

        await this.doLlmGeneration(messages, metaInfo, nextStep, shouldBypassSynth);

        if (this.useLlmToDetermineHangup && !this.turnBasedConversation) {
            const llmAgent = this.tools["llm_agent"] as {
                checkForCompletion: (msgs: unknown[], prompt: string) => Promise<[Record<string, unknown>, Record<string, unknown>]>
            };
            const [completionRes, metadata] = await llmAgent.checkForCompletion(messages, this.checkForCompletionPrompt!);

            const shouldHangup = typeof completionRes === "object"
                ? String(completionRes["hangup"] ?? "").toLowerCase() === "yes"
                : false;

            (this.llmLatencies["other_latencies"] as unknown[]).push({
                type: "hangup_check",
                latency_ms: (metadata["latency_ms"] as number | undefined) ?? null,
                model: this.checkForCompletionLlm,
                provider: "openai",
                sequence_id: metaInfo["sequence_id"],
            });

            if (shouldHangup) {
                if (this.hangupTriggered || this.conversationEnded) {
                    logger.info("Hangup already triggered or conversation ended, skipping");
                    return;
                }
                this.hangupDetail = "llm_prompted_hangup";
                await this.processCallHangup();
                return;
            }
        }

        this.llmProcessedRequestIds.add(this.currentRequestId!);
    }

    async processCallHangup(): Promise<void> {
        if (this.hangupTriggered || this.conversationEnded) {
            logger.info("processCallHangup: already in progress, skipping");
            return;
        }

        this.hangupTriggered = true;
        this.hangupTriggeredAt = Date.now() / 1000;
        const message = !this.voicemailDetected ? this.callHangupMessage : "";

        if (!message?.trim()) {
            this.hangupMessageQueued = false;
            await this.processEndOfConversation();
        } else {
            this.hangupMessageQueued = true;
            await this.waitForCurrentMessage();
            await this.cleanupDownstreamTasks();
            const metaInfo: Record<string, unknown> = {
                io: (this.tools["output"] as { getProvider: () => string }).getProvider(),
                request_id: uuidv4(),
                cached: false,
                sequence_id: -1,
                format: "pcm",
                message_category: "agent_hangup",
                end_of_llm_stream: true,
            };
            await this.synthesize(createWsDataPacket({ data: message, metaInfo }));
        }
    }

    private async listenLlmInputQueue(): Promise<void> {
        logger.info(`Starting listening to LLM queue`);
        while (true) {
            try {
                const wsDataPacket = await this.queues["llm"]!.get() as Record<string, unknown>;
                logger.info(`ws_data_packet ${JSON.stringify(wsDataPacket)}`);
                const metaInfo = this.getUpdatedMetaInfo(wsDataPacket["meta_info"] as Record<string, unknown>);
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: "<beginning_of_stream>", metaInfo })
                );
                await this.runLlmTask(createWsDataPacket({ data: wsDataPacket["data"], metaInfo }));
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: "<end_of_stream>", metaInfo })
                );
            } catch (e) {
                logger.error(`Something went wrong with LLM queue ${e}`);
                break;
            }
        }
    }

    private async runLlmTask(message: unknown): Promise<void> {
        const [sequence, metaInfo] = this.extractSequenceAndMeta(message);
        try {
            if (this.isExtractionTask() || this.isSummarizationTask()) {
                await this.processFollowupTask(message);
            } else if (this.isConversationTask()) {
                await this.processConversationTask(message as Record<string, unknown>, sequence ?? 0, metaInfo ?? {});
            } else {
                logger.error(`unsupported task type: ${this.taskConfig["task_type"]}`);
            }
            this.llmTask = null;
        } catch (e) {
            logger.error(`Something went wrong in llm: ${e}`);
            this.responseInPipeline = false;
        }
    }

    // ── Transcriber ──────────────────────────────────────────────────────────

    async processTranscriberRequest(metaInfo: Record<string, unknown>): Promise<number> {
        const requestId = metaInfo["request_id"] as string | undefined;
        if (requestId && (!this.currentRequestId || this.currentRequestId !== requestId)) {
            this.previousRequestId = this.currentRequestId;
            this.currentRequestId = requestId;
        }
        const sequence = (metaInfo["sequence"] as number | undefined) ?? 0;
        if (this.previousRequestId == null) {
            // first message
        } else if (!this.llmProcessedRequestIds.has(this.previousRequestId)) {
            this.llmRejectedRequestIds.add(this.previousRequestId);
        }
        return sequence;
    }

    private shouldCheckVoicemail(transcriberMessage: string, isFinal = true): boolean {
        if (!this.voicemailDetectionEnabled || this.voicemailDetected) return false;
        if (this.voicemailCheckTask != null) { logger.info("Voicemail check already in progress"); return false; }

        const currentTime = Date.now() / 1000;
        if (this.voicemailDetectionStartTime == null) {
            this.voicemailDetectionStartTime = currentTime;
        }

        const timeElapsed = currentTime - this.voicemailDetectionStartTime;
        if (timeElapsed > this.voicemailDetectionDuration) return false;

        if (!isFinal) {
            const timeSinceLastCheck = this.voicemailLastCheckTime != null
                ? currentTime - this.voicemailLastCheckTime : Infinity;
            if (timeSinceLastCheck < this.voicemailCheckInterval) return false;
            const wordCount = transcriberMessage.trim().split(" ").length;
            if (wordCount < this.voicemailMinTranscriptLength) return false;
        }
        return true;
    }

    private triggerVoicemailCheck(transcriberMessage: string, metaInfo: Record<string, unknown>, isFinal = true): void {
        if (!this.shouldCheckVoicemail(transcriberMessage, isFinal)) return;
        this.voicemailLastCheckTime = Date.now() / 1000;
        try {
            this.voicemailCheckTask = this.voicemailCheckBackgroundTask(transcriberMessage, metaInfo, isFinal);
        } catch (e) {
            logger.error(`Error starting voicemail check background task: ${e}`);
        }
    }

    private async voicemailCheckBackgroundTask(
        transcriberMessage: string,
        metaInfo: Record<string, unknown>,
        _isFinal: boolean
    ): Promise<void> {
        try {
            const llmAgent = this.tools["llm_agent"] as { checkForVoicemail?: (msg: string, prompt: string) => Promise<[Record<string, unknown>, Record<string, unknown>]> };
            if ("checkForVoicemail" in (llmAgent ?? {})) {
                const [voicemailResult, metadata] = await llmAgent.checkForVoicemail!(transcriberMessage, this.voicemailDetectionPrompt);
                const isVoicemail = typeof voicemailResult === "object"
                    ? String(voicemailResult["is_voicemail"] ?? "").toLowerCase() === "yes" : false;

                (this.llmLatencies["other_latencies"] as unknown[]).push({
                    type: "voicemail_check",
                    latency_ms: (metadata["latency_ms"] as number | undefined) ?? null,
                    model: this.voicemailLlm,
                    provider: "openai",
                    sequence_id: metaInfo["sequence_id"] ?? null,
                });

                if (isVoicemail) {
                    logger.info(`Voicemail detected! Message: ${transcriberMessage}`);
                    this.voicemailDetected = true;
                    this.hangupDetail = "voicemail_detected";
                    await this.handleVoicemailDetected();
                }
            } else {
                logger.warn("Voicemail detection enabled but llm_agent doesn't support checkForVoicemail");
            }
        } catch (e) {
            logger.error(`Error during background voicemail detection: ${e}`);
        }
    }

    private async handleVoicemailDetected(): Promise<void> {
        logger.info("Handling voicemail detection - ending call");
        await this.processCallHangup();
    }

    private async handleTranscriberOutput(
        nextTask: string,
        transcriberMessage: string,
        metaInfo: Record<string, unknown>
    ): Promise<void> {
        const inputTool = this.tools["input"] as { welcomeMessagePlayed: () => boolean; isAudioBeingPlayedToUser: () => boolean };
        if (!inputTool.welcomeMessagePlayed() && this.conversationHistory.length > 2) {
            logger.info(`Welcome message is playing while spoken: ${transcriberMessage}`);
            return;
        }

        if (this.conversationHistory.isDuplicateUser(transcriberMessage)) {
            logger.info(`Skipping duplicate transcript: ${transcriberMessage}`);
            return;
        }

        this.triggerVoicemailCheck(transcriberMessage, metaInfo, true);

        if (this.voicemailDetected) {
            logger.info("Voicemail already detected - skipping normal processing");
            return;
        }

        await this.languageDetector.collectTranscript(transcriberMessage);

        if (this.responseInPipeline && nextTask === "llm") {
            this.conversationHistory.popUnheardResponses();
            this.invalidateResponseChain();
            const original = transcriberMessage;
            transcriberMessage = this.conversationHistory.popAndMergeUser(transcriberMessage);
            if (transcriberMessage !== original) logger.info(`Merged transcript: ${transcriberMessage}`);
            await this.cleanupDownstreamTasks();
        }

        this.conversationHistory.appendUser(transcriberMessage);
        convertToRequestLog({
            message: transcriberMessage, metaInfo, model: (this.taskConfig["tools_config"] as Record<string, unknown>)["transcriber"]
                ? ((this.taskConfig["tools_config"] as Record<string, unknown>)["transcriber"] as Record<string, unknown>)["provider"] as string : "", runId: this.runId
        });

        if (nextTask === "llm") {
            logger.info("Running llm Tasks");
            metaInfo["origin"] = "transcriber";
            const transcriberPackage = createWsDataPacket({ data: transcriberMessage, metaInfo });

            if (this.llmTask != null) {
                logger.info("Cancelling existing LLM task for new speech_final");
                this.llmTask = null;
                this.interruption_manager.invalidatePendingResponses();
                this.interruption_manager.revalidateSequenceId(metaInfo["sequence_id"] as number);
            }

            this.responseInPipeline = true;
            this.llmTask = this.runLlmTask(transcriberPackage);
        } else if (nextTask === "synthesizer") {
            this.synthesizerTasks.push(this.synthesize(createWsDataPacket({ data: transcriberMessage, metaInfo })));
        }
    }

    private async listenTranscriber(): Promise<void> {
        let tempTranscriberMessage = "";
        try {
            while (true) {
                const message = await this.transcriberOutputQueue.get() as Record<string, unknown>;
                logger.info(`Message from the transcriber class ${JSON.stringify(message)}`);

                if (this.hangupTriggered) {
                    if (message["data"] === "transcriber_connection_closed") {
                        logger.info("Transcriber connection has been closed");
                        const mi = message["meta_info"] as Record<string, unknown> | null;
                        this.transcriberDuration += (mi?.["transcriber_duration"] as number | undefined) ?? 0;
                        break;
                    }
                    continue;
                }

                if (this.stream) {
                    this.setCallDetails(message);
                    const metaInfo = message["meta_info"] as Record<string, unknown>;
                    const sequence = await this.processTranscriberRequest(metaInfo);
                    const nextTask = this.getNextStep(sequence, "transcriber");
                    let interimTranscriptLen = 0;

                    if (message["data"] === "speech_started") {
                        if ((this.tools["input"] as { welcomeMessagePlayed: () => boolean }).welcomeMessagePlayed()) {
                            logger.info("User has started speaking");
                        }
                    } else if (
                        typeof message["data"] === "object" &&
                        (message["data"] as Record<string, unknown>)["type"] === "interim_transcript_received"
                    ) {
                        this.timeSinceLastSpokenHumanWord = Date.now() / 1000;
                        const content = (message["data"] as Record<string, unknown>)["content"] as string;
                        if (tempTranscriberMessage === content) continue;
                        tempTranscriberMessage = content;

                        if (!(this.tools["input"] as { welcomeMessagePlayed: () => boolean }).welcomeMessagePlayed()) continue;

                        interimTranscriptLen += content.trim().split(" ").length;

                        if (this.interruption_manager.shouldTriggerInterruption(
                            interimTranscriptLen,
                            content,
                            (this.tools["input"] as { isAudioBeingPlayedToUser: () => boolean }).isAudioBeingPlayedToUser() || this.responseInPipeline,
                            (this.tools["input"] as { welcomeMessagePlayed: () => boolean }).welcomeMessagePlayed(),
                        )) {
                            logger.info("Condition for interruption hit");
                            this.interruption_manager.onUserSpeechStarted();
                            this.interruption_manager.onInterruptionTriggered();
                            (this.tools["input"] as { updateIsAudioBeingPlayed: (v: boolean) => void }).updateIsAudioBeingPlayed(false);
                            await this.cleanupDownstreamTasks();
                        } else if (!(this.tools["input"] as { isAudioBeingPlayedToUser: () => boolean }).isAudioBeingPlayedToUser() && (this.tools["input"] as { welcomeMessagePlayed: () => boolean }).welcomeMessagePlayed()) {
                            this.interruption_manager.onUserSpeechStarted();
                            const hasPendingResponse = this.interruption_manager.hasPendingResponses();
                            const timeSinceUtteranceEnd = this.interruption_manager.getTimeSinceUtteranceEnd();
                            const withinGracePeriod = timeSinceUtteranceEnd !== -1 && timeSinceUtteranceEnd < this.incrementalDelay && this.history.length > 2;
                            if (hasPendingResponse && withinGracePeriod && interimTranscriptLen > this.numberOfWordsForInterruption) {
                                logger.info("User continuation detected, canceling pending response");
                                this.interruption_manager.resetUtteranceEndTime();
                                await this.cleanupDownstreamTasks();
                            }
                        } else if ((this.tools["input"] as { isAudioBeingPlayedToUser: () => boolean }).isAudioBeingPlayedToUser() || this.responseInPipeline) {
                            logger.info(`Ignoring transcript: ${content.trim()}`);
                            continue;
                        } else {
                            this.interruption_manager.onUserSpeechStarted();
                        }

                        this.interruption_manager.updateRequiredDelay(this.history.length);
                        this.interruption_manager.onInterimTranscriptReceived();

                        if (this.voicemailDetectionEnabled && !this.voicemailDetected) {
                            this.triggerVoicemailCheck(content, metaInfo, false);
                        }
                        this.llmResponseGenerated = false;

                    } else if (
                        typeof message["data"] === "object" &&
                        (message["data"] as Record<string, unknown>)["type"] === "transcript"
                    ) {
                        const transcriptContent = (message["data"] as Record<string, unknown>)["content"] as string;
                        const wordCount = transcriptContent.trim().split(" ").length;

                        if (this.interruption_manager.isFalseInterruption(
                            wordCount,
                            transcriptContent,
                            (this.tools["input"] as { isAudioBeingPlayedToUser: () => boolean }).isAudioBeingPlayedToUser() || this.responseInPipeline,
                            (this.tools["input"] as { welcomeMessagePlayed: () => boolean }).welcomeMessagePlayed(),
                        )) {
                            logger.info(`Ignoring false interruption: ${transcriptContent}`);
                            this.interruption_manager.onUserSpeechEnded(false);
                            continue;
                        }

                        this.interruption_manager.onUserSpeechEnded();
                        tempTranscriberMessage = "";

                        if (this.outputTask == null) {
                            this.outputTask = this.processOutputLoop();
                        }
                        this.interruption_manager.resetDelayForSpeechFinal(this.history.length);
                        const updatedMeta = this.getUpdatedMetaInfo(metaInfo);
                        await this.handleTranscriberOutput(nextTask, transcriptContent, updatedMeta);

                    } else if (
                        typeof message["data"] === "object" &&
                        (message["data"] as Record<string, unknown>)["type"] === "speech_ended"
                    ) {
                        logger.info("Received speech_ended notification");
                        this.interruption_manager.onUserSpeechEnded(false);
                        tempTranscriberMessage = "";

                    } else if (message["data"] === "transcriber_connection_closed") {
                        logger.info("Transcriber connection has been closed");
                        const mi = message["meta_info"] as Record<string, unknown> | null;
                        this.transcriberDuration += (mi?.["transcriber_duration"] as number | undefined) ?? 0;
                        break;
                    }
                } else {
                    if (message["data"] === "transcriber_connection_closed") {
                        const mi = message["meta_info"] as Record<string, unknown> | null;
                        this.transcriberDuration += (mi?.["transcriber_duration"] as number | undefined) ?? 0;
                        break;
                    }
                    await this.processHttpTranscription(message);
                }
            }
        } catch (e) {
            logger.error(`Error in transcriber ${e}`);
        }
    }

    private async processHttpTranscription(message: Record<string, unknown>): Promise<void> {
        const metaInfo = this.getUpdatedMetaInfo(message["meta_info"] as Record<string, unknown>);
        const sequence = (message["meta_info"] as Record<string, unknown>)["sequence"] as number ?? metaInfo["sequence_id"] as number;
        const nextTask = this.getNextStep(sequence, "transcriber");
        const mi = message["meta_info"] as Record<string, unknown>;
        this.transcriberDuration += (mi["transcriber_duration"] as number | undefined) ?? 0;
        await this.handleTranscriberOutput(nextTask, message["data"] as string, metaInfo);
    }

    // ── Synthesizer ──────────────────────────────────────────────────────────

    private enqueueChunk(chunk: Buffer, i: number, numberOfChunks: number, metaInfo: Record<string, unknown>): void {
        metaInfo["chunk_id"] = i;
        const copiedMetaInfo = structuredClone(metaInfo);
        if (i === 0 && metaInfo["is_first_chunk"]) {
            copiedMetaInfo["is_first_chunk_of_entire_response"] = true;
        }
        if (i === numberOfChunks - 1 && (metaInfo["sequence_id"] === -1 || metaInfo["end_of_synthesizer_stream"])) {
            copiedMetaInfo["is_final_chunk_of_entire_response"] = true;
            delete copiedMetaInfo["is_first_chunk_of_entire_response"];
        }
        if (copiedMetaInfo["message_category"] === "agent_welcome_message") {
            copiedMetaInfo["is_first_chunk_of_entire_response"] = true;
            copiedMetaInfo["is_final_chunk_of_entire_response"] = true;
        }
        this.bufferedOutputQueue.putNowait(createWsDataPacket({ data: chunk, metaInfo: copiedMetaInfo }));
    }

    isSequenceIdInCurrentIds(sequenceId: number): boolean {
        return this.interruption_manager.isValidSequence(sequenceId);
    }

    private async listenSynthesizer(): Promise<void> {
        const allTextToBeSynthesized: string[] = [];
        try {
            while (!this.conversationEnded) {
                logger.info("Listening to synthesizer");
                try {
                    const synth = this.tools["synthesizer"] as { generate: () => AsyncGenerator<Record<string, unknown>>; getSleepTime: () => number; cleanup: () => Promise<void> };
                    for await (const message of synth.generate()) {
                        const metaInfo = message["meta_info"] as Record<string, unknown>;
                        const currentText = (metaInfo["text"] as string) ?? "";
                        let writeToLog = false;
                        if (!allTextToBeSynthesized.includes(currentText)) {
                            allTextToBeSynthesized.push(currentText);
                            writeToLog = true;
                        }
                        const isFirstMessage = metaInfo["is_first_message"] as boolean | undefined;
                        const sequenceId = metaInfo["sequence_id"] as number | undefined;

                        if (isFirstMessage || (!this.conversationEnded && this.interruption_manager.isValidSequence(sequenceId!))) {
                            if (this.stream) {
                                const outputTool = this.tools["output"] as { processInChunks: (yc: boolean) => boolean };
                                if (outputTool.processInChunks(this.yieldChunks)) {
                                    const data = message["data"] as Buffer;
                                    const numberOfChunks = Math.ceil(data.length / this.outputChunkSize);
                                    let idx = 0;
                                    for (const chunk of yieldChunksFromMemory(data, this.outputChunkSize)) {
                                        this.enqueueChunk(chunk, idx++, numberOfChunks, metaInfo);
                                    }
                                } else {
                                    this.bufferedOutputQueue.putNowait(message);
                                }
                            } else {
                                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(message);
                            }

                            if (writeToLog) {
                                convertToRequestLog({ message: currentText, metaInfo, component: "synthesizer", direction: "response", model: this.synthesizerProvider, isCached: metaInfo["is_cached"] as boolean, engine: (this.tools["synthesizer"] as { getEngine: () => string }).getEngine(), runId: this.runId });
                            }
                        } else {
                            logger.info(`Skipping message with sequence_id: ${sequenceId}`);
                        }

                        await new Promise((r) => setTimeout(r, synth.getSleepTime() * 1_000));
                    }
                } catch (e) {
                    logger.error(`Error in synthesizer: ${e}`);
                    break;
                }
            }
        } catch (e) {
            logger.error(`Unexpected error in listenSynthesizer: ${e}`);
        } finally {
            await (this.tools["synthesizer"] as { cleanup: () => Promise<void> }).cleanup();
        }
    }

    private async synthesize(message: unknown): Promise<void> {
        const msg = message as Record<string, unknown>;
        const metaInfo = msg["meta_info"] as Record<string, unknown>;
        const text = msg["data"] as string;
        metaInfo["type"] = "audio";
        metaInfo["synthesizer_start_time"] = Date.now() / 1000;

        try {
            if (!this.conversationEnded &&
                (metaInfo["is_first_message"] || this.interruption_manager.isValidSequence(metaInfo["sequence_id"] as number))
            ) {
                if (metaInfo["is_md5_hash"]) {
                    await this.sendPreprocessedAudio(metaInfo, text);
                } else if (this.synthesizerProvider in SUPPORTED_SYNTHESIZER_MODELS) {
                    convertToRequestLog({ message: text, metaInfo, component: "synthesizer", direction: "request", model: this.synthesizerProvider, engine: (this.tools["synthesizer"] as { getEngine: () => string }).getEngine(), runId: this.runId });
                    if (metaInfo["cached"] === true) {
                        convertToRequestLog({ message: text, metaInfo, component: "synthesizer", direction: "response", model: this.synthesizerProvider, isCached: true, engine: (this.tools["synthesizer"] as { getEngine: () => string }).getEngine(), runId: this.runId });
                        await this.sendPreprocessedAudio(metaInfo, getMd5Hash(text));
                    } else {
                        this.synthesizerCharacters += text.length;
                        await (this.tools["synthesizer"] as { push: (msg: unknown) => Promise<void> }).push(message);
                    }
                }
            } else {
                logger.info(`${metaInfo["sequence_id"]} is not a valid sequence id and hence not synthesizing`);
            }
        } catch (e) {
            logger.error(`Error in synthesizer: ${e}`);
        }
    }

    private async sendPreprocessedAudio(metaInfo: Record<string, unknown>, text: string): Promise<void> {
        const copiedMeta = structuredClone(metaInfo);
        try {
            const outputProvider = (this.taskConfig["tools_config"] as Record<string, unknown>)["output"]
                ? ((this.taskConfig["tools_config"] as Record<string, unknown>)["output"] as Record<string, unknown>)["provider"] as string
                : "default";

            if (this.turnBasedConversation || outputProvider === "default") {
                const audioChunk = await getRawAudioBytes({
                    filename: text,
                    assistantId: this.assistantId as string,
                    audioFormat: copiedMeta["format"] as string,
                    local: this.isLocal,
                    isLocation: true,
                });
                copiedMeta["format"] = ((this.taskConfig["tools_config"] as Record<string, unknown>)["output"] as Record<string, unknown>)["format"];
                copiedMeta["end_of_synthesizer_stream"] = true;
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(createWsDataPacket({ data: audioChunk, metaInfo: copiedMeta }));
            } else {
                if (copiedMeta["message_category"] === "filler") {
                    const audio = await getRawAudioBytes({
                        filename: `${this.fuelPresetDirectory}/${text}.wav`,
                        local: true,
                        isLocation: true,
                    });
                    if (!this.turnBasedConversation && outputProvider !== "default") {
                        if (audio === null) {
                            logger.error("Audio file not found");
                            return;
                        }
                        const resampled = await resample(audio, 8000, { format: "wav" });
                        const audioChunk = wavBytesToPcm(resampled);
                        copiedMeta["format"] = "pcm";
                        copiedMeta["end_of_synthesizer_stream"] = true;
                        this.bufferedOutputQueue.putNowait(createWsDataPacket({ data: audioChunk, metaInfo: copiedMeta }));
                    }
                } else {
                    let audioChunk: Buffer | null = this.preloadedWelcomeAudio;
                    if (copiedMeta["text"] === "") audioChunk = null;
                    if (!this.bufferedOutputQueue.empty()) this.bufferedOutputQueue = new AsyncQueue();

                    copiedMeta["format"] = "pcm";
                    if (copiedMeta["message_category"] === "agent_welcome_message") {
                        if (audioChunk == null) {
                            copiedMeta["cached"] = false;
                            await this.synthesize(createWsDataPacket({ data: copiedMeta["text"], metaInfo: copiedMeta }));
                            return;
                        } else {
                            copiedMeta["is_first_chunk"] = true;
                        }
                    }
                    copiedMeta["end_of_synthesizer_stream"] = true;

                    if (audioChunk != null) {
                        let i = 0;
                        const numberOfChunks = Math.ceil(audioChunk.length / 100_000_000);
                        for (const chunk of yieldChunksFromMemory(audioChunk, 100_000_000)) {
                            this.enqueueChunk(chunk, i++, numberOfChunks, copiedMeta);
                        }
                    }
                }
            }
        } catch (e) {
            logger.error(`Something went wrong in sendPreprocessedAudio: ${e}`);
        }
    }

    // ── Output loop ──────────────────────────────────────────────────────────

    private async processOutputLoop(): Promise<void> {
        try {
            while (true) {
                const inputTool = this.tools["input"] as { welcomeMessagePlayed: () => boolean; isAudioBeingPlayedToUser: () => boolean; updateIsAudioBeingPlayed: (v: boolean) => void };
                if (inputTool.welcomeMessagePlayed()) {
                    const [shouldDelay, sleepDuration] = this.interruption_manager.shouldDelayOutput(inputTool.welcomeMessagePlayed()) as [boolean, number];
                    if (shouldDelay) {
                        await new Promise((r) => setTimeout(r, sleepDuration * 1_000));
                        continue;
                    }
                }

                const message = await this.bufferedOutputQueue.get() as Record<string, unknown>;
                const metaInfo = message["meta_info"] as Record<string, unknown>;

                if (metaInfo["end_of_conversation"]) {
                    await this.processEndOfConversation();
                }

                const sequenceId = metaInfo["sequence_id"] as number;
                let shouldContinueOuterLoop = false;

                while (true) {
                    const status = this.interruption_manager.getAudioSendStatus(sequenceId, this.history.length);

                    if (status === "SEND") {
                        inputTool.updateIsAudioBeingPlayed(true);
                        this.responseInPipeline = false;
                        await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(message);
                        try {
                            const data = message["data"] as Buffer;
                            const format = Number(metaInfo["format"]);
                            const duration = calculateAudioDuration(data.length, this.samplingRate, format);
                            (this.conversationRecording["output"] as unknown[]).push({ data, start_time: Date.now() / 1000, duration });
                        } catch { /* ignore */ }
                        break;
                    } else if (status === "BLOCK") {
                        logger.info(`Audio blocked: discarding message (sequence_id=${sequenceId})`);
                        if (message["data"] === Buffer.from([0x00])) {
                            await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(message);
                        }
                        shouldContinueOuterLoop = true;
                        break;
                    } else {
                        // WAIT
                        await new Promise((r) => setTimeout(r, 50));
                    }
                }

                if (shouldContinueOuterLoop) continue;

                if ((metaInfo["end_of_llm_stream"] || metaInfo["end_of_synthesizer_stream"]) &&
                    metaInfo["message_category"] !== "is_user_online_message") {
                    this.askedIfUserIsStillThere = false;
                }
            }
        } catch (e) {
            logger.error(`Error in processOutputLoop: ${e}`);
        }
    }

    private async checkForCompletion(): Promise<void> {
        logger.info("Starting task to check for completion");
        while (true) {
            await new Promise((r) => setTimeout(r, 2_000));

            if (this.isWebBasedCall) {
                const maxDuration = parseInt((this.taskConfig["task_config"] as Record<string, unknown>)["call_terminate"] as string);
                if (Date.now() / 1000 - this.startTime >= maxDuration) {
                    logger.info("Hanging up for web call as max time reached");
                    await this.processEndOfConversation(true);
                    this.hangupDetail = "web_call_max_duration_reached";
                    break;
                }
            }

            if (this.lastTransmittedTimestamp === 0) continue;

            if (this.hangupTriggered) {
                if (this.conversationEnded) { logger.info("Call hangup completed"); break; }
                if (this.hangupTriggeredAt != null) {
                    const timeSince = Date.now() / 1000 - this.hangupTriggeredAt;
                    if (timeSince > this.hangupMarkEventTimeout) {
                        logger.warn(`Hangup mark event not received within ${this.hangupMarkEventTimeout}s, forcing conversation end`);
                        if ("output" in this.tools) (this.tools["output"] as { setHangupSent: () => void }).setHangupSent();
                        await this.processEndOfConversation();
                        break;
                    }
                }
                continue;
            }

            if ((this.tools["input"] as { isAudioBeingPlayedToUser: () => boolean }).isAudioBeingPlayedToUser()) continue;

            const timeSinceLastAiWord = Date.now() / 1000 - this.lastTransmittedTimestamp;
            const timeSinceUserLastSpoke = this.timeSinceLastSpokenHumanWord > 0
                ? Date.now() / 1000 - this.timeSinceLastSpokenHumanWord : Infinity;

            if (this.hangConversationAfter > 0 && timeSinceLastAiWord > this.hangConversationAfter && timeSinceUserLastSpoke > this.hangConversationAfter) {
                logger.info(`Both AI and user silent for ${this.hangConversationAfter}s - hanging up`);
                this.hangupDetail = "inactivity_timeout";
                await this.processCallHangup();
                break;
            } else if (timeSinceLastAiWord > this.triggerUserOnlineMessageAfter && !this.askedIfUserIsStillThere && timeSinceUserLastSpoke > this.triggerUserOnlineMessageAfter) {
                logger.info("Asking if user is still there");
                this.askedIfUserIsStillThere = true;

                if (this.checkIfUserOnline) {
                    const detectedLang = this.languageDetector.dominantLanguage;
                    const userOnlineMessage = selectMessageByLanguage(this.checkUserOnlineMessageConfig as string | Record<string, string>, detectedLang);
                    const format = this.shouldRecord ? "wav" : "pcm";
                    const io = this.shouldRecord ? "default" : (this.tools["output"] as { getProvider: () => string }).getProvider();
                    const metaInfo: Record<string, unknown> = { io, request_id: uuidv4(), cached: false, sequence_id: -1, format, message_category: "is_user_online_message", end_of_llm_stream: true };
                    await this.synthesize(createWsDataPacket({ data: userOnlineMessage, metaInfo }));
                    this.history.push({ role: "assistant", content: userOnlineMessage });
                }
                await (this.tools["output"] as { handleInterruption: () => Promise<void> }).handleInterruption();
            }
        }
    }

    private async checkForBackchanneling(): Promise<void> {
        const { randomInt } = await import("crypto");
        while (true) {
            const userSpeakingDuration = this.interruption_manager.getUserSpeakingDuration();
            if (this.interruption_manager.isUserSpeaking() && userSpeakingDuration > this.backchannelingStartDelay) {
                const filename = this.filenames[randomInt(this.filenames.length)]!;
                logger.info(`Sending backchanneling audio: ${filename}`);
                let audio = await getRawAudioBytes({
                    filename: `${this.backchannelingAudios}/${filename}`,
                    local: true,
                    isLocation: true
                });
                if (!this.turnBasedConversation) {
                    if (audio === null) {
                        logger.error("Backchanneling audio not found");
                        return
                    }
                    const resampled = await resample(audio, 8000, { format: "wav" });
                    audio = wavBytesToPcm(resampled);
                }
                await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(
                    createWsDataPacket({ data: audio, metaInfo: this.getUpdatedMetaInfo() })
                );
            }
            await new Promise((r) => setTimeout(r, this.backchannelingMessageGap * 1_000));
        }
    }

    private async firstMessage(timeout = 10.0): Promise<void> {
        logger.info("Executing the first message task");
        try {
            if (this.isWebBasedCall) {
                logger.info("Sending agent welcome message for web based call");
                const text = this.kwargs["agent_welcome_message"] as string | undefined ?? null;
                const metaInfo: Record<string, unknown> = {
                    io: "default", message_category: "agent_welcome_message",
                    stream_sid: this.streamSid, request_id: uuidv4(), cached: false,
                    sequence_id: -1,
                    format: (this.taskConfig["tools_config"] as Record<string, unknown>)["output"]
                        ? ((this.taskConfig["tools_config"] as Record<string, unknown>)["output"] as Record<string, unknown>)["format"] : "mp3",
                    text, end_of_llm_stream: true,
                };
                this.streamSidTs = Date.now();
                await this.synthesize(createWsDataPacket({ data: text, metaInfo }));
                return;
            }

            const startTime = Date.now() / 1000;
            while (true) {
                const elapsed = Date.now() / 1000 - startTime;
                if (elapsed > timeout) {
                    await this.processEndOfConversation();
                    logger.warn("Timeout reached while waiting for stream_sid");
                    break;
                }

                if (!this.streamSid && !this.defaultIo) {
                    const streamSid = (this.tools["input"] as { getStreamSid: () => string | null }).getStreamSid();
                    if (streamSid != null) {
                        this.streamSidTs = Date.now();
                        logger.info(`Got stream sid, sending first message ${streamSid}`);
                        this.streamSid = streamSid;
                        const text = this.kwargs["agent_welcome_message"] as string | undefined ?? null;
                        const metaInfo: Record<string, unknown> = {
                            io: (this.tools["output"] as { getProvider: () => string }).getProvider(),
                            message_category: "agent_welcome_message", stream_sid: streamSid,
                            request_id: uuidv4(), cached: true, sequence_id: -1,
                            format: (this.taskConfig["tools_config"] as Record<string, unknown>)["output"]
                                ? ((this.taskConfig["tools_config"] as Record<string, unknown>)["output"] as Record<string, unknown>)["format"] : "pcm",
                            text, end_of_llm_stream: true,
                        };
                        if (this.turnBasedConversation) {
                            metaInfo["type"] = "text";
                            await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(createWsDataPacket({ data: "<beginning_of_stream>", metaInfo }));
                            await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(createWsDataPacket({ data: text, metaInfo }));
                            await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(createWsDataPacket({ data: "<end_of_stream>", metaInfo }));
                        } else {
                            await this.synthesize(createWsDataPacket({ data: text, metaInfo }));
                        }
                        break;
                    } else {
                        await new Promise((r) => setTimeout(r, 10));
                    }
                } else if (this.defaultIo) {
                    break;
                }
            }
        } catch (e) {
            logger.error(`Exception in firstMessage: ${e}`);
        }
    }

    private async startTransmittingAmbientNoise(): Promise<void> {
        try {
            let audio = await getRawAudioBytes({
                filename: `${process.env.AMBIENT_NOISE_PRESETS_DIR}/${this.soundtrack}`,
                local: true,
                isLocation: true,
            });
            if (audio === null) {
                // Handle the case where audio is null
                // For example, you can log an error or return early
                logger.error('Failed to get audio data');
                return;
            }
            audio = await resample(audio, this.samplingRate, { format: "wav" });
            const outputProvider = (this.taskConfig["tools_config"] as Record<string, unknown>)["output"]
                ? ((this.taskConfig["tools_config"] as Record<string, unknown>)["output"] as Record<string, unknown>)["provider"] as string : "default";
            if (outputProvider in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS) {
                audio = wavBytesToPcm(audio);
            }
            const format = this.shouldRecord ? "wav" : "pcm";
            const io = this.shouldRecord ? "default" : (this.tools["output"] as { getProvider: () => string }).getProvider();
            const metaInfo: Record<string, unknown> = { io, message_category: "ambient_noise", stream_sid: this.streamSid, request_id: uuidv4(), cached: true, type: "audio", sequence_id: -1, format };

            while (true) {
                for (const chunk of yieldChunksFromMemory(audio, this.outputChunkSize * 2)) {
                    const isContentPlaying = (this.tools["input"] as { isAudioBeingPlayedToUser: () => boolean }).isAudioBeingPlayedToUser();
                    if (!isContentPlaying) {
                        await (this.tools["output"] as { handle: (m: unknown) => Promise<void> }).handle(createWsDataPacket({ data: chunk, metaInfo }));
                    }
                    await new Promise((r) => setTimeout(r, 500));
                }
            }
        } catch (e) {
            logger.error(`Something went wrong while transmitting ambient noise: ${e}`);
        }
    }

    async handleInitEvent(initMetaData: Record<string, unknown>): Promise<void> {
        try {
            logger.info(`handleInitEvent triggered with metadata = ${JSON.stringify(initMetaData)}`);
            (this.contextData!["recipient_data"] as Record<string, unknown>)["context_data"]
                ? Object.assign(this.contextData!["recipient_data"] as Record<string, unknown>, (initMetaData["context_data"] as Record<string, unknown>))
                : Object.assign(this.contextData!["recipient_data"] as Record<string, unknown>, initMetaData["context_data"] as Record<string, unknown>);

            this.prompts["system_prompt"] = updatePromptWithContext(this.prompts["system_prompt"] as string, this.contextData);

            if (this.systemPrompt["content"]) {
                let sp = this.systemPrompt["content"];
                sp = updatePromptWithContext(sp, this.contextData);
                this.systemPrompt["content"] = sp;
                this.conversationHistory.updateSystemPrompt(sp);
            }

            if (this.callHangupMessageConfig && this.contextData) {
                if (typeof this.callHangupMessageConfig === "object") {
                    const map = this.callHangupMessageConfig as Record<string, string>;
                    this.callHangupMessageConfig = Object.fromEntries(
                        Object.entries(map).map(([lang, msg]) => [lang, updatePromptWithContext(msg, this.contextData)])
                    );
                } else {
                    this.callHangupMessageConfig = updatePromptWithContext(this.callHangupMessageConfig as string, this.contextData);
                }
            }

            let welcomeMsg = (this.kwargs["agent_welcome_message"] as string | undefined) ?? "";
            welcomeMsg = updatePromptWithContext(welcomeMsg, this.contextData);
            this.kwargs["agent_welcome_message"] = welcomeMsg;
            if (this.conversationHistory.length === 2 && welcomeMsg) {
                this.conversationHistory.updateWelcomeMessage(welcomeMsg);
            }

            await (this.tools["output"] as { sendInitAcknowledgement: () => Promise<void> }).sendInitAcknowledgement();
            this.firstMessageTask = this.firstMessage();
        } catch (e) {
            logger.error(`Error occurred in handleInitEvent: ${e}`);
        }
    }

    private async handleAccumulatedMessage(): Promise<void> {
        logger.info("Setting up handleAccumulatedMessage function");
        while (true) {
            if ((this.tools["input"] as { welcomeMessagePlayed: () => boolean }).welcomeMessagePlayed()) {
                logger.info("Welcome message has been played");
                this.firstMessagePassingTime = Date.now() / 1000;
                if (this.transcribedMessage.length) {
                    logger.info(`Sending accumulated transcribed message: ${this.transcribedMessage}`);
                    await this.sendFirstMessage(this.transcribedMessage);
                    this.transcribedMessage = "";
                }
                break;
            }
            await new Promise((r) => setTimeout(r, 100));
        }
        this.handleAccumulatedMessageTask = null;
    }

    private async sendFirstMessage(message: string): Promise<void> {
        const metaInfo = this.getUpdatedMetaInfo();
        const sequence = (metaInfo["sequence"] as number | undefined) ?? 0;
        const nextTask = this.getNextStep(sequence, "transcriber");
        await this.handleTranscriberOutput(nextTask, message, metaInfo);
        this.interruption_manager.setFirstInterimForImmediateResponse();
    }

    // ── Main run ─────────────────────────────────────────────────────────────

    async run(): Promise<Record<string, unknown>> {
        try {
            if (this.isConversationTask()) {
                logger.info("started running");
                const tasks: Promise<void>[] = [];

                if (this.turnBasedConversation) {
                    this.firstMessageTask = this.firstMessage();
                }

                if (!this.turnBasedConversation) {
                    this.firstMessagePassingTime = null;
                    this.handleAccumulatedMessageTask = this.handleAccumulatedMessage();
                }

                if ("transcriber" in this.tools) {
                    tasks.push(this.listenTranscriber());
                    this.transcriber_task = (this.tools["transcriber"] as { run: () => Promise<void> }).run();
                }

                if (this.turnBasedConversation) {
                    this.llmQueueTask = this.listenLlmInputQueue();
                }

                if ("synthesizer" in this.tools && !this.turnBasedConversation) {
                    this.synthesizerTask = this.listenSynthesizer();
                }

                this.outputTask = this.processOutputLoop();

                if (!this.turnBasedConversation || this.enforceStreaming) {
                    this.hangupTask = this.checkForCompletion();
                    if (this.shouldBackchannel) {
                        this.backchannelingTask = this.checkForBackchanneling();
                    }
                    if (this.ambientNoise) {
                        this.ambientNoiseTask = this.startTransmittingAmbientNoise();
                    }
                }

                await Promise.all(tasks);

                if (this.generatePreciseTranscript) {
                    const hasPendingMarks = Object.keys(this.markEventMetaData.getMarkEventMetaData).length > 0;
                    const hasResponseHeard = !!(this.tools["input"] as { responseHeardByUser?: string }).responseHeardByUser;
                    if (hasPendingMarks || hasResponseHeard) {
                        await this.syncHistory(Object.entries(this.markEventMetaData.getMarkEventMetaData), Date.now() / 1000);
                    }
                    (this.tools["input"] as { resetResponseHeardByUser: () => void }).resetResponseHeardByUser();
                }

                logger.info("Conversation completed");
                this.conversationEnded = true;

            } else {
                try {
                    if ((this.taskConfig["task_type"] as string) === "webhook") {
                        await this.processFollowupTask();
                    } else {
                        await this.runLlmTask(this.inputParameters!);
                    }
                } catch (e) {
                    logger.error(`Could not do llm call: ${e}`);
                    throw e;
                }
            }
        } catch (e) {
            logger.error(`Exception in task manager run: ${e}`);
            await this.handleCancellation(`Exception occurred ${e}`);
            throw e;
        } finally {
            const tasksToCancel: Promise<unknown>[] = [];

            if ("synthesizer" in this.tools && this.synthesizerTask != null) {
                tasksToCancel.push((this.tools["synthesizer"] as { cleanup: () => Promise<void> }).cleanup());
                tasksToCancel.push(processTaskCancellation(this.synthesizerTask, "synthesizer_task"));
                if (this.synthesizerMonitorTask) tasksToCancel.push(processTaskCancellation(this.synthesizerMonitorTask, "synthesizer_monitor_task"));
            }

            if ("transcriber" in this.tools) {
                tasksToCancel.push((this.tools["transcriber"] as { cleanup: () => Promise<void> }).cleanup());
                if (this.transcriber_task) tasksToCancel.push(processTaskCancellation(this.transcriber_task, "transcriber_task"));
            }

            let output: Record<string, unknown> = {};

            if (this.isConversationTask()) {
                this.transcriberLatencies["connection_latency_ms"] = (this.tools["transcriber"] as { connectionTime: number }).connectionTime;
                this.synthesizerLatencies["connection_latency_ms"] = (this.tools["synthesizer"] as { connectionTime: number }).connectionTime;
                this.transcriberLatencies["turn_latencies"] = (this.tools["transcriber"] as { turnLatencies: unknown[] }).turnLatencies;
                this.synthesizerLatencies["turn_latencies"] = (this.tools["synthesizer"] as { turnLatencies: unknown[] }).turnLatencies;

                if (this.languageDetector?.latencyData) {
                    (this.llmLatencies["other_latencies"] as unknown[]).push(this.languageDetector.latencyData);
                }

                const welcomeMessageSentTs = (this.tools["output"] as { getWelcomeMessageSentTs: () => number | null }).getWelcomeMessageSentTs();

                output = {
                    messages: this.history,
                    conversation_time: Date.now() / 1000 - this.startTime,
                    label_flow: this.labelFlow,
                    call_sid: this.callSid,
                    stream_sid: this.streamSid,
                    transcriber_duration: this.transcriberDuration,
                    synthesizer_characters: (this.tools["synthesizer"] as { getSynthesizedCharacters: () => number }).getSynthesizedCharacters(),
                    ended_by_assistant: this.endedByAssistant,
                    latency_dict: {
                        llm_latencies: this.llmLatencies,
                        transcriber_latencies: this.transcriberLatencies,
                        synthesizer_latencies: this.synthesizerLatencies,
                        rag_latencies: this.ragLatencies,
                        routing_latencies: this.routingLatencies,
                        welcome_message_sent_ts: null as number | null,
                        stream_sid_ts: null as number | null,
                    },
                    hangup_detail: this.hangupDetail,
                    recording_url: "",
                };

                try {
                    if (welcomeMessageSentTs) {
                        (output["latency_dict"] as Record<string, unknown>)["welcome_message_sent_ts"] = welcomeMessageSentTs - this.conversationStartInitTs;
                    }
                    if (this.streamSidTs) {
                        (output["latency_dict"] as Record<string, unknown>)["stream_sid_ts"] = this.streamSidTs - this.conversationStartInitTs;
                    }
                } catch (e) {
                    logger.error(`error in logging audio latency ts: ${e}`);
                }

                for (const [taskRef, taskName] of [
                    [this.outputTask, "output_task"],
                    [this.hangupTask, "hangup_task"],
                    [this.backchannelingTask, "backchanneling_task"],
                    [this.ambientNoiseTask, "ambient_noise_task"],
                    [this.firstMessageTask, "first_message_task"],
                    [this.dtmfTask, "dtmf_task"],
                    [this.voicemailCheckTask, "voicemail_check_task"],
                    [this.handleAccumulatedMessageTask, "handle_accumulated_message_task"],
                ] as [Promise<void> | null, string][]) {
                    if (taskRef) tasksToCancel.push(processTaskCancellation(taskRef, taskName));
                }

                if (this.shouldRecord) {
                    const audioBuffer = Buffer.concat([
                        this.conversationRecording.input.data,
                        ...this.conversationRecording.output
                    ]);
                    await saveAudioFileToS3(
                        audioBuffer,
                        `${this.runId}.wav`,
                        this.samplingRate
                    );
                }
            } else {
                output = this.inputParameters ?? {};
                const taskType = this.taskConfig["task_type"] as string;
                if (taskType === "extraction") {
                    output = { extracted_data: this.extractedData, task_type: "extraction", latency_dict: { llm_latencies: this.llmLatencies } };
                } else if (taskType === "summarization") {
                    output = { summary: this.summarizedData, task_type: "summarization", latency_dict: { llm_latencies: this.llmLatencies } };
                } else if (taskType === "webhook") {
                    output = { status: this.webhookResponse, task_type: "webhook" };
                }
            }

            await Promise.allSettled(tasksToCancel);
            return output;
        }
    }

    async handleCancellation(message: string): Promise<void> {
        try {
            logger.info(message);
        } catch (e) {
            logger.error(`Error in handleCancellation: ${e}`);
        }
    }
}