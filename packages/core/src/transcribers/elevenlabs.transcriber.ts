import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket, timestampMs, type WsDataPacket } from "../helper/utils";
import { BaseTranscriber } from "./base";

const logger = configureLogger("elevenlabsTranscriber");

export class ElevenLabsTranscriber extends BaseTranscriber {
    private endpointing: string;
    private vadSilenceThresholdSecs: number;
    private language: string;
    private stream: boolean;
    private provider: string;
    private senderTask: Promise<void> | null = null;
    private utteranceTimeoutTask: Promise<void> | null = null;
    private model: string;
    private samplingRate: number;
    private encoding: string;
    private apiKey: string;
    private elevenlabsHost: string;
    private transcriberOutputQueue: { put: (data: unknown) => Promise<void> };
    private transcriptionTask: Promise<void> | null = null;
    private transcriptionCursor = 0.0;
    private interruptionSignalled = false;
    private audioSubmitted = false;
    private audioSubmissionTime: number | null = null;
    private numFrames = 0;
    private connectionStartTime: number | null = null;
    private audioFrameDuration = 0.0;
    private connectedViaDashboard: boolean;

    // ElevenLabs-specific
    private commitStrategy: string;
    private includeTimestamps: boolean;
    private includeLanguageDetection: boolean;
    private vadThreshold: number;
    private minSpeechDurationMs: number;
    private minSilenceDurationMs: number;

    // Turn state
    private currMessage = "";
    private finalizedTranscript = "";
    private finalTranscript = "";
    private isTranscriptSentForProcessing = false;
    private currentTurnStartTime: number | null = null;
    private currentTurnId: string | number | null = null;
    private websocketConnection: WebSocket | null = null;
    private connectionAuthenticated = false;
    private speechStartTime: number | null = null;
    private speechEndTime: number | null = null;
    private currentTurnInterimDetails: Record<string, unknown>[] = [];
    private audioFrameTimestamps: [number, number, number][] = [];
    private turnCounter = 0;
    private lastAudioSendTime: number | null = null;
    private lastInterimTime: number | null = null;
    private interimTimeout: number;

    // Receiver plumbing
    private messageResolvers: ((msg: string | null) => void)[] = [];
    private pendingMessages: (string | null)[] = [];
    private utteranceAbort = new AbortController();

    constructor(opts: {
        telephonyProvider: string;
        inputQueue?: { get: () => Promise<unknown> } | null;
        model?: string;
        stream?: boolean;
        language?: string;
        endpointing?: string;
        samplingRate?: string;
        encoding?: string;
        outputQueue: { put: (data: unknown) => Promise<void> };
        commitStrategy?: string;
        includeTimestamps?: boolean;
        includeLanguageDetection?: boolean;
        transcriberKey?: string;
        enforceStreaming?: boolean;
        interimTimeout?: number;
        vadThreshold?: number;
        minSpeechDurationMs?: number;
        minSilenceDurationMs?: number;
        [key: string]: unknown;
    }) {
        super(opts.inputQueue ?? null);

        this.endpointing = opts.endpointing ?? "400";
        const rawVad = parseInt(this.endpointing, 10) / 1000.0;
        this.vadSilenceThresholdSecs = Math.max(0.3, Math.min(3.0, rawVad));

        this.language = opts.language ?? "en";
        this.stream = opts.stream ?? true;
        this.provider = opts.telephonyProvider;
        this.model = opts.model ?? "scribe_v2_realtime";
        this.samplingRate = 16000;
        this.encoding = opts.encoding ?? "linear16";
        this.apiKey = (opts.transcriberKey as string | undefined) ?? process.env.ELEVENLABS_API_KEY!;
        this.elevenlabsHost = process.env.ELEVENLABS_API_HOST ?? "api.elevenlabs.io";
        this.transcriberOutputQueue = opts.outputQueue;
        this.connectedViaDashboard = (opts.enforceStreaming as boolean | undefined) ?? true;

        this.commitStrategy = opts.commitStrategy ?? "vad";
        this.includeTimestamps = opts.includeTimestamps ?? true;
        this.includeLanguageDetection = opts.includeLanguageDetection ?? true;

        this.vadThreshold = Math.max(0.1, Math.min(0.9, (opts.vadThreshold as number | undefined) ?? 0.5));
        this.minSpeechDurationMs = Math.max(50, Math.min(2000, (opts.minSpeechDurationMs as number | undefined) ?? 150));
        this.minSilenceDurationMs = Math.max(50, Math.min(2000, (opts.minSilenceDurationMs as number | undefined) ?? 300));
        this.interimTimeout = (opts.interimTimeout as number | undefined) ?? 5.0;
        this.metaInfo = {};
    }

    // ------------------------------------------------------------------
    // URL builder
    // ------------------------------------------------------------------

    private getElevenlabsWsUrl(): string {
        this.audioFrameDuration = 0.5;
        let audioFormat = "pcm_16000";

        if (["twilio", "exotel", "plivo", "vobiz"].includes(this.provider)) {
            this.encoding = this.provider === "twilio" ? "mulaw" : "linear16";
            this.samplingRate = 8000;
            this.audioFrameDuration = 0.2;
            audioFormat = this.provider === "twilio" ? "ulaw_8000" : "pcm_8000";
        } else if (this.provider === "web_based_call") {
            this.encoding = "linear16";
            this.samplingRate = 16000;
            this.audioFrameDuration = 0.256;
            audioFormat = "pcm_16000";
        } else if (!this.connectedViaDashboard) {
            this.encoding = "linear16";
            this.samplingRate = 16000;
            audioFormat = "pcm_16000";
        }

        if (this.provider === "playground") {
            this.samplingRate = 8000;
            this.audioFrameDuration = 0.0;
            audioFormat = "pcm_8000";
        }

        const params = new URLSearchParams({
            model_id: this.model,
            language_code: this.language,
            audio_format: audioFormat,
            commit_strategy: this.commitStrategy,
            vad_silence_threshold_secs: String(this.vadSilenceThresholdSecs),
            vad_threshold: String(this.vadThreshold),
            min_speech_duration_ms: String(this.minSpeechDurationMs),
            min_silence_duration_ms: String(this.minSilenceDurationMs),
            include_timestamps: this.includeTimestamps ? "true" : "false",
            include_language_detection: this.includeLanguageDetection ? "true" : "false",
        });

        const url = `wss://${this.elevenlabsHost}/v1/speech-to-text/realtime?${params.toString()}`;
        logger.info(
            `ElevenLabs WS params - language: ${this.language}, audio_format: ${audioFormat}, ` +
            `vad_threshold: ${this.vadThreshold}, min_speech_ms: ${this.minSpeechDurationMs}, ` +
            `min_silence_ms: ${this.minSilenceDurationMs}, vad_silence_secs: ${this.vadSilenceThresholdSecs}`
        );
        return url;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private findAudioSendTimestamp(audioPosition: number): number | null {
        for (const [frameStart, frameEnd, sendTs] of this.audioFrameTimestamps) {
            if (frameStart <= audioPosition && audioPosition <= frameEnd) return sendTs;
        }
        return null;
    }

    private resetTurnState(): void {
        this.speechStartTime = null;
        this.speechEndTime = null;
        this.lastInterimTime = null;
        this.currentTurnInterimDetails = [];
        this.currentTurnStartTime = null;
        this.currentTurnId = null;
        this.finalTranscript = "";
        this.isTranscriptSentForProcessing = false;
    }

    // ------------------------------------------------------------------
    // Force-finalize
    // ------------------------------------------------------------------

    private async forceFinalizeUtterance(): Promise<void> {
        let transcriptToSend = this.finalTranscript.trim();

        if (!transcriptToSend && this.currentTurnInterimDetails.length) {
            transcriptToSend =
                (this.currentTurnInterimDetails[this.currentTurnInterimDetails.length - 1]!["transcript"] as string) ?? "";
            logger.info(`Using last interim as fallback: ${transcriptToSend}`);
        }

        if (!transcriptToSend) {
            logger.warn("No transcript available to force-finalize");
            this.resetTurnState();
            return;
        }

        try {
            const [firstToFinal, lastToFinal] = this.calculateInterimToFinalLatencies(
                this.currentTurnInterimDetails as { received_at?: number }[]
            );
            this.turnLatencies.push({
                turn_id: this.currentTurnId,
                sequence_id: this.currentTurnId,
                interim_details: this.currentTurnInterimDetails,
                first_interim_to_final_ms: firstToFinal,
                last_interim_to_final_ms: lastToFinal,
                force_finalized: true,
            });
        } catch (e) {
            logger.error(`Error building turn latencies: ${e}`);
        }

        logger.info(`Force-finalized transcript: ${transcriptToSend}`);
        await this.pushToTranscriberQueue(
            createWsDataPacket({ data: { type: "transcript", content: transcriptToSend, force_finalized: true }, metaInfo: this.metaInfo! })
        );
        this.resetTurnState();
    }

    // ------------------------------------------------------------------
    // Utterance timeout monitor
    // ------------------------------------------------------------------

    private async monitorUtteranceTimeout(): Promise<void> {
        try {
            while (!this.utteranceAbort.signal.aborted) {
                await new Promise((r) => setTimeout(r, 1_000));
                if (this.utteranceAbort.signal.aborted) break;

                if (
                    this.lastInterimTime !== null &&
                    !this.isTranscriptSentForProcessing &&
                    (this.finalTranscript.trim() || this.currentTurnInterimDetails.length)
                ) {
                    const elapsed = Date.now() / 1000 - this.lastInterimTime;
                    if (elapsed > this.interimTimeout) {
                        logger.warn(
                            `Interim timeout: ${elapsed.toFixed(1)}s. Force-finalizing turn ${this.currentTurnId}`
                        );
                        await this.forceFinalizeUtterance();
                    }
                }
            }
        } catch (e) {
            logger.error(`Error in monitorUtteranceTimeout: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Toggle / cleanup
    // ------------------------------------------------------------------

    async toggleConnection(): Promise<void> {
        this.connectionOn = false;
        this.utteranceAbort.abort();

        if (this.websocketConnection) {
            try {
                this.websocketConnection.close();
                logger.info("Websocket connection closed successfully");
            } catch (e) {
                logger.error(`Error closing websocket: ${e}`);
            } finally {
                this.websocketConnection = null;
                this.connectionAuthenticated = false;
            }
        }
    }

    override async cleanup(): Promise<void> {
        logger.info("Cleaning up ElevenLabs transcriber resources");
        this.utteranceAbort.abort();

        if (this.websocketConnection) {
            try { this.websocketConnection.close(); } catch (e) {
                logger.error(`Error closing ElevenLabs websocket: ${e}`);
            }
            this.websocketConnection = null;
            this.connectionAuthenticated = false;
        }
        logger.info("ElevenLabs transcriber cleanup complete");
    }

    // ------------------------------------------------------------------
    // EOS check
    // ------------------------------------------------------------------

    private async checkAndProcessEndOfStream(
        wsDataPacket: Record<string, unknown>,
        ws: WebSocket
    ): Promise<boolean> {
        const metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
        if (metaInfo["eos"] === true) {
            try { ws.close(); } catch (e) {
                logger.debug(`Error closing websocket on EOS: ${e}`);
            }
            return true;
        }
        return false;
    }

    getMetaInfo(): Record<string, unknown> {
        return this.metaInfo ?? {};
    }

    // ------------------------------------------------------------------
    // Sender stream
    // ------------------------------------------------------------------

    private async senderStream(ws: WebSocket): Promise<void> {
        try {
            while (true) {
                const wsDataPacket = (await (this.inputQueue as { get: () => Promise<Record<string, unknown>> }).get());

                if (!this.audioSubmitted) {
                    this.metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                    this.audioSubmitted = true;
                    this.audioSubmissionTime = Date.now() / 1000;
                    this.currentRequestId = BaseTranscriber.generateRequestId();
                    this.metaInfo["request_id"] = this.currentRequestId;
                    if (!this.currentTurnStartTime) {
                        this.currentTurnStartTime = timestampMs();
                        this.currentTurnId = (this.metaInfo["turn_id"] ?? this.metaInfo["request_id"]) as string;
                    }
                }

                const eos = await this.checkAndProcessEndOfStream(wsDataPacket, ws);
                if (eos) break;

                const frameStart = this.numFrames * this.audioFrameDuration;
                const frameEnd = (this.numFrames + 1) * this.audioFrameDuration;
                this.audioFrameTimestamps.push([frameStart, frameEnd, timestampMs()]);
                this.numFrames++;

                try {
                    const audioData = wsDataPacket["data"] as Buffer;
                    const audioB64 = audioData.toString("base64");
                    ws.send(JSON.stringify({
                        message_type: "input_audio_chunk",
                        audio_base_64: audioB64,
                        sample_rate: this.samplingRate,
                        commit: false,
                    }));
                    this.lastAudioSendTime = timestampMs();
                } catch (e) {
                    logger.error(`Error sending data to websocket: ${e}`);
                    break;
                }
            }
        } catch (e) {
            logger.error(`Error in senderStream: ${e}`);
            throw e;
        }
    }

    // ------------------------------------------------------------------
    // Receiver async generator
    // ------------------------------------------------------------------

    private async *receiver(ws: WebSocket): AsyncGenerator<WsDataPacket> {
        ws.on("message", (raw) => {
            const msg = raw.toString();
            const resolver = this.messageResolvers.shift();
            if (resolver) resolver(msg);
            else this.pendingMessages.push(msg);
        });
        ws.once("close", () => {
            const resolver = this.messageResolvers.shift();
            if (resolver) resolver(null);
            else this.pendingMessages.push(null);
        });
        ws.once("error", () => {
            const resolver = this.messageResolvers.shift();
            if (resolver) resolver(null);
            else this.pendingMessages.push(null);
        });

        const nextMessage = (): Promise<string | null> =>
            new Promise((resolve) => {
                const pending = this.pendingMessages.shift();
                if (pending !== undefined) resolve(pending);
                else this.messageResolvers.push(resolve);
            });

        try {
            while (true) {
                const raw = await nextMessage();
                if (raw === null) break;

                try {
                    const msg = JSON.parse(raw) as Record<string, unknown>;
                    const msgType = (msg["message_type"] as string) ?? "";

                    if (this.connectionStartTime === null) {
                        this.connectionStartTime = Date.now() / 1000 - this.numFrames * this.audioFrameDuration;
                    }

                    // --- session_started ---
                    if (msgType === "session_started") {
                        logger.info(`ElevenLabs session started: ${msg["session_id"]}`);
                        this.connectionAuthenticated = true;

                        // --- partial_transcript ---
                    } else if (msgType === "partial_transcript") {
                        const transcript = ((msg["text"] as string) ?? "").trim();

                        if (transcript) {
                            if (!this.currentTurnId) {
                                this.turnCounter++;
                                this.currentTurnId = this.turnCounter;
                                this.speechStartTime = timestampMs();
                                this.currentTurnInterimDetails = [];
                                logger.info(`Starting new turn: ${this.currentTurnId}`);
                                yield createWsDataPacket({ data: "speech_started", metaInfo: this.metaInfo! });
                            }

                            let latencyMs: number | null = null;
                            if (this.lastAudioSendTime !== null) {
                                latencyMs = Math.round((timestampMs() - this.lastAudioSendTime) * 100000) / 100000;
                            }

                            this.currentTurnInterimDetails.push({
                                transcript, is_final: false, received_at: Date.now() / 1000, latency_ms: latencyMs,
                            });
                            this.lastInterimTime = Date.now() / 1000;
                            logger.info(`Partial transcript: ${transcript} (latency: ${latencyMs}ms)`);

                            this.finalTranscript = transcript;
                            if (this.isTranscriptSentForProcessing) this.isTranscriptSentForProcessing = false;

                            yield createWsDataPacket({
                                data: { type: "interim_transcript_received", content: transcript },
                                metaInfo: this.metaInfo!,
                            });
                        }

                        // --- committed_transcript ---
                    } else if (msgType === "committed_transcript") {
                        const transcript = ((msg["text"] as string) ?? "").trim();
                        logger.info(`Committed transcript: ${transcript}`);

                        // Skip when include_timestamps is on — we process committed_transcript_with_timestamps instead
                        if (this.includeTimestamps) continue;

                        if (transcript && !this.isTranscriptSentForProcessing) {
                            try {
                                const [firstToFinal, lastToFinal] = this.calculateInterimToFinalLatencies(
                                    this.currentTurnInterimDetails as { received_at?: number }[]
                                );
                                this.turnLatencies.push({
                                    turn_id: this.currentTurnId,
                                    sequence_id: this.currentTurnId,
                                    interim_details: this.currentTurnInterimDetails,
                                    first_interim_to_final_ms: firstToFinal,
                                    last_interim_to_final_ms: lastToFinal,
                                });
                                this.speechStartTime = null;
                                this.speechEndTime = null;
                                this.currentTurnInterimDetails = [];
                                this.currentTurnStartTime = null;
                                this.currentTurnId = null;
                                this.finalTranscript = "";
                                this.isTranscriptSentForProcessing = false;
                            } catch (e) {
                                logger.error(`Error in committed_transcript handling: ${e}`);
                            }

                            yield createWsDataPacket({
                                data: { type: "transcript", content: transcript },
                                metaInfo: this.metaInfo!,
                            });
                        }

                        // --- committed_transcript_with_timestamps ---
                    } else if (msgType === "committed_transcript_with_timestamps") {
                        const transcript = ((msg["text"] as string) ?? "").trim();
                        const words = (msg["words"] as Record<string, unknown>[]) ?? [];
                        const detectedLanguage = msg["language_code"] as string | undefined;
                        logger.info(`Committed transcript with timestamps: ${transcript} (${words.length} words)`);

                        if (transcript && !this.isTranscriptSentForProcessing) {
                            try {
                                if (words.length && this.audioFrameTimestamps.length) {
                                    for (const wordObj of words) {
                                        if (typeof wordObj === "object" && "end" in wordObj) {
                                            const audioPos = wordObj["end"] as number;
                                            const audioSentAt = this.findAudioSendTimestamp(audioPos);
                                            if (audioSentAt !== null) {
                                                wordObj["latency_ms"] = Math.round((timestampMs() - audioSentAt) * 100000) / 100000;
                                            }
                                        }
                                    }
                                }

                                const [firstToFinal, lastToFinal] = this.calculateInterimToFinalLatencies(
                                    this.currentTurnInterimDetails as { received_at?: number }[]
                                );
                                this.turnLatencies.push({
                                    turn_id: this.currentTurnId,
                                    sequence_id: this.currentTurnId,
                                    interim_details: this.currentTurnInterimDetails,
                                    first_interim_to_final_ms: firstToFinal,
                                    last_interim_to_final_ms: lastToFinal,
                                    words,
                                    detected_language: detectedLanguage,
                                });

                                this.resetTurnState();
                            } catch (e) {
                                logger.error(`Error in committed_transcript_with_timestamps handling: ${e}`);
                            }

                            yield createWsDataPacket({
                                data: { type: "transcript", content: transcript },
                                metaInfo: this.metaInfo!,
                            });
                        }

                        // --- input_error ---
                    } else if (msgType === "input_error") {
                        logger.warn(`ElevenLabs input error: ${msg["error"] ?? "Unknown"}`);

                        // --- unaccepted_terms ---
                    } else if (msgType === "unaccepted_terms") {
                        logger.error(`ElevenLabs terms not accepted: ${msg["error"]}`);
                        logger.error("Please accept terms at https://elevenlabs.io/app/product-terms");
                        break;

                        // --- error ---
                    } else if (msgType === "error") {
                        logger.error(`ElevenLabs error: ${msg["error"] ?? "Unknown"}`);

                    } else {
                        logger.debug(`Unknown message type: ${msgType} - ${JSON.stringify(msg)}`);
                    }
                } catch (e) {
                    console.error(e);
                    logger.error(`Error processing message: ${e}`);
                    this.interruptionSignalled = false;
                }
            }
        } catch (e) {
            console.error(e);
        }
    }

    // ------------------------------------------------------------------
    // Queue
    // ------------------------------------------------------------------

    async pushToTranscriberQueue(dataPacket: unknown): Promise<void> {
        await this.transcriberOutputQueue.put(dataPacket);
    }

    // ------------------------------------------------------------------
    // Connect
    // ------------------------------------------------------------------

    private async elevenlabsConnect(): Promise<WebSocket> {
        const wsUrl = this.getElevenlabsWsUrl();
        logger.info(`Attempting to connect to ElevenLabs websocket: ${wsUrl}`);

        return new Promise<WebSocket>((resolve, reject) => {
            const timer = setTimeout(
                () => reject(new ConnectionError("Timeout connecting to ElevenLabs")),
                10_000
            );

            const ws = new WebSocket(wsUrl, {
                headers: { "xi-api-key": this.apiKey },
            });

            ws.once("open", () => {
                clearTimeout(timer);
                this.websocketConnection = ws;
                logger.info("Successfully connected to ElevenLabs websocket");
                resolve(ws);
            });

            ws.once("error", (err) => {
                clearTimeout(timer);
                reject(new ConnectionError(`ElevenLabs connection error: ${err.message}`));
            });
        });
    }

    // ------------------------------------------------------------------
    // Transcribe (main entry)
    // ------------------------------------------------------------------

    private async transcribe(): Promise<void> {
        let elevenlabsWs: WebSocket | null = null;

        try {
            const start = timestampMs();

            try {
                elevenlabsWs = await this.elevenlabsConnect();
            } catch (e) {
                logger.error(`Failed to establish ElevenLabs connection: ${e}`);
                await this.toggleConnection();
                return;
            }

            if (!this.connectionTime) {
                this.connectionTime = Math.round(timestampMs() - start);
            }

            if (this.stream) {
                this.senderTask = this.senderStream(elevenlabsWs);
                this.utteranceTimeoutTask = this.monitorUtteranceTimeout();

                try {
                    for await (const message of this.receiver(elevenlabsWs)) {
                        if (this.connectionOn) {
                            await this.pushToTranscriberQueue(message);
                        } else {
                            logger.info("Closing the ElevenLabs connection");
                            break;
                        }
                    }
                } catch (e) {
                    logger.error(`Error during streaming: ${e}`);
                    throw e;
                }

                await Promise.allSettled([this.senderTask, this.utteranceTimeoutTask]);
            }
        } catch (e) {
            logger.error(`Unexpected error in transcribe: ${e}`);
            await this.toggleConnection();
        } finally {
            this.utteranceAbort.abort();

            if (elevenlabsWs) {
                try {
                    elevenlabsWs.close();
                    logger.info("ElevenLabs websocket closed in finally block");
                } catch (e) {
                    logger.error(`Error closing websocket in finally: ${e}`);
                }
                this.websocketConnection = null;
                this.connectionAuthenticated = false;
            }

            await this.pushToTranscriberQueue(
                createWsDataPacket({ data: "transcriber_connection_closed", metaInfo: this.metaInfo ?? {} })
            );
        }
    }

    async run(): Promise<void> {
        try {
            this.transcriptionTask = this.transcribe();
            await this.transcriptionTask;
        } catch (e) {
            logger.error(`Error starting transcription: ${e}`);
        }
    }
}

class ConnectionError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "ConnectionError";
    }
}