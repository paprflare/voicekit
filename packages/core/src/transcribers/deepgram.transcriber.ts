import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket, timestampMs, type WsDataPacket } from "../helper/utils";
import { BaseTranscriber } from "./base";

const logger = configureLogger("deepgramTranscriber");

const TELEPHONY_PROVIDERS = new Set(["twilio", "plivo", "exotel", "vobiz", "sip_trunk"]);

export class DeepgramTranscriber extends BaseTranscriber {
    private endpointing: string;
    private language: string;
    private stream: boolean;
    private provider: string;
    private heartbeatTask: Promise<void> | null = null;
    private senderTask: Promise<void> | null = null;
    private utteranceTimeoutTask: Promise<void> | null = null;
    private model: string;
    private samplingRate: number;
    private encoding: string;
    private apiKey: string;
    private deepgramHost: string;
    private transcriberOutputQueue: { put: (data: unknown) => Promise<void> };
    private transcriptionTask: Promise<void> | null = null;
    private keywords: string | null;
    private transcriptionCursor = 0.0;
    private interruptionSignalled = false;
    private apiUrl: string;

    private audioSubmitted = false;
    private audioSubmissionTime: number | null = null;
    private numFrames = 0;
    private connectionStartTime: number | null = null;
    private processInterimResults: string;
    private audioFrameDuration = 0.0;
    private connectedViaDashboard: boolean;

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
    private lastInterimTime: number | null = null;
    private interimTimeout: number;

    // Receiver plumbing
    private messageResolvers: ((msg: string | null) => void)[] = [];
    private pendingMessages: (string | null)[] = [];

    // Utterance timeout abort
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
        keywords?: string | null;
        processInterimResults?: string;
        transcriberKey?: string;
        enforceStreaming?: boolean;
        interimTimeout?: number;
        [key: string]: unknown;
    }) {
        super(opts.inputQueue ?? null);

        this.endpointing = opts.endpointing ?? "400";
        this.language = opts.language ?? "en";
        this.stream = opts.stream ?? true;
        this.provider = opts.telephonyProvider;
        this.model = opts.model ?? "nova-2";
        this.samplingRate = typeof opts.samplingRate === "string"
            ? parseInt(opts.samplingRate, 10)
            : (opts.samplingRate as number | undefined) ?? 16000;
        this.encoding = opts.encoding ?? "linear16";
        this.apiKey = (opts.transcriberKey as string | undefined) ?? process.env.DEEPGRAM_AUTH_TOKEN!;
        this.deepgramHost = process.env.DEEPGRAM_HOST ?? "api.deepgram.com";
        this.transcriberOutputQueue = opts.outputQueue;
        this.keywords = opts.keywords ?? null;
        this.processInterimResults = opts.processInterimResults ?? "true";
        this.connectedViaDashboard = (opts.enforceStreaming as boolean | undefined) ?? true;
        this.interimTimeout = (opts.interimTimeout as number | undefined) ?? 5.0;
        this.metaInfo = {};
        this.apiUrl = `https://${this.deepgramHost}/v1/listen?model=${this.model}&filler_words=true&language=${this.language}`;

        if (this.keywords) {
            const kwStr = this.keywords.split(",").map((k) => `keywords=${k}`).join("&");
            this.apiUrl = `${this.apiUrl}&${kwStr}`;
        }
    }

    // ------------------------------------------------------------------
    // URL builder
    // ------------------------------------------------------------------

    private getDeepgramWsUrl(): string {
        const params: Record<string, string | number | boolean> = {
            model: this.model,
            filler_words: "true",
            language: this.language,
            vad_events: "true",
            endpointing: this.endpointing,
            interim_results: "true",
            utterance_end_ms: parseInt(this.endpointing, 10) < 1000 ? "1000" : this.endpointing,
        };

        this.audioFrameDuration = 0.5;

        if (TELEPHONY_PROVIDERS.has(this.provider)) {
            if (this.provider !== "sip_trunk") {
                this.encoding = this.provider === "twilio" ? "mulaw" : "linear16";
                this.samplingRate = 8000;
            }
            this.audioFrameDuration = 0.2;
            params["encoding"] = this.encoding;
            params["sample_rate"] = this.samplingRate;
            params["channels"] = "1";

            if (this.provider === "sip_trunk") {
                logger.info(`[SIP-TRUNK] Deepgram configured: encoding=${this.encoding} sample_rate=${this.samplingRate}`);
            }
        } else if (this.provider === "web_based_call") {
            params["encoding"] = "linear16";
            params["sample_rate"] = 16000;
            params["channels"] = "1";
            this.samplingRate = 16000;
            this.audioFrameDuration = 0.256;
        } else if (!this.connectedViaDashboard) {
            params["encoding"] = "linear16";
            params["sample_rate"] = 16000;
            params["channels"] = "1";
        }

        if (this.provider === "playground") {
            this.samplingRate = 8000;
            this.audioFrameDuration = 0.0;
        }

        if (!this.language.includes("en")) params["language"] = this.language;

        if (this.keywords && this.keywords.split(",").length > 0) {
            if (this.model.startsWith("nova-3")) {
                params["keyterm"] = this.keywords.split(",").join("&keyterm=");
                if (this.language !== "en") delete params["keyterm"];
            } else {
                params["keywords"] = this.keywords.split(",").join("&keywords=");
            }
        }

        const protocol = process.env.DEEPGRAM_HOST_PROTOCOL ?? "wss";
        const qs = Object.entries(params).map(([k, v]) => `${k}=${v}`).join("&");
        return `${protocol}://${this.deepgramHost}/v1/listen?${qs}`;
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

    private setTranscriptionCursor(data: Record<string, unknown>): number {
        if ("start" in data && "duration" in data) {
            this.transcriptionCursor = (data["start"] as number) + (data["duration"] as number);
            logger.info(`Transcription cursor: ${this.transcriptionCursor}`);
        } else {
            logger.warn("Missing start/duration in Deepgram message");
        }
        return this.transcriptionCursor;
    }

    private resetTurnState(): void {
        this.speechStartTime = null;
        this.speechEndTime = null;
        this.lastInterimTime = null;
        this.currentTurnInterimDetails = [];
        this.currentTurnStartTime = null;
        this.currentTurnId = null;
        this.finalTranscript = "";
        this.isTranscriptSentForProcessing = true;
    }

    // ------------------------------------------------------------------
    // Force-finalize stuck utterance
    // ------------------------------------------------------------------

    private async forceFinalizeUtterance(): Promise<void> {
        let transcriptToSend = this.finalTranscript.trim();

        if (!transcriptToSend && this.currentTurnInterimDetails.length) {
            transcriptToSend = (this.currentTurnInterimDetails[this.currentTurnInterimDetails.length - 1]!["transcript"] as string) ?? "";
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

        logger.info(`Force-finalized transcript after timeout: ${transcriptToSend}`);
        await this.pushToTranscriberQueue(
            createWsDataPacket({
                data: { type: "transcript", content: transcriptToSend, force_finalized: true },
                metaInfo: this.metaInfo!,
            })
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
                            `Interim timeout: No finalization for ${elapsed.toFixed(1)}s. ` +
                            `Force-finalizing turn ${this.currentTurnId}`
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
    // Heartbeat
    // ------------------------------------------------------------------

    private async sendHeartbeat(ws: WebSocket): Promise<void> {
        try {
            while (true) {
                await new Promise((r) => setTimeout(r, 5_000));
                if (ws.readyState !== WebSocket.OPEN) break;
                try {
                    ws.send(JSON.stringify({ type: "KeepAlive" }));
                } catch (e) {
                    logger.error(`Error sending heartbeat: ${e}`);
                    break;
                }
            }
        } catch (e) {
            logger.error(`Error in sendHeartbeat: ${e}`);
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
        logger.info("Cleaning up Deepgram transcriber resources");
        this.utteranceAbort.abort();

        if (this.websocketConnection) {
            try { this.websocketConnection.close(); } catch (e) {
                logger.error(`Error closing Deepgram websocket: ${e}`);
            }
            this.websocketConnection = null;
            this.connectionAuthenticated = false;
        }
        logger.info("Deepgram transcriber cleanup complete");
    }

    // ------------------------------------------------------------------
    // HTTP transcription
    // ------------------------------------------------------------------

    private async getHttpTranscription(audioData: Buffer): Promise<WsDataPacket> {
        this.currentRequestId = BaseTranscriber.generateRequestId();
        this.metaInfo!["request_id"] = this.currentRequestId;

        const res = await fetch(this.apiUrl, {
            method: "POST",
            headers: {
                Authorization: `Token ${this.apiKey}`,
                "Content-Type": "audio/webm",
            },
            body: Buffer.from(audioData).buffer,
        });

        const data = (await res.json()) as Record<string, unknown>;
        const results = data["results"] as Record<string, unknown>;
        const channels = (results["channels"] as Record<string, unknown>[])[0]!;
        const alternatives = (channels["alternatives"] as Record<string, unknown>[])[0]!;
        const transcript = (alternatives["transcript"] as string) ?? "";
        const metadata = data["metadata"] as Record<string, unknown>;
        this.metaInfo!["transcriber_duration"] = metadata["duration"] ?? 0;
        return createWsDataPacket({ data: transcript, metaInfo: this.metaInfo! });
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
            await this._close({ send: (d) => Promise.resolve(ws.send(d)) }, { type: "CloseStream" });
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Sender (HTTP non-streaming)
    // ------------------------------------------------------------------

    private async *sender(): AsyncGenerator<WsDataPacket> {
        try {
            while (true) {
                const wsDataPacket = (await (this.inputQueue as { get: () => Promise<Record<string, unknown>> }).get());

                if (!this.audioSubmitted) {
                    this.audioSubmitted = true;
                    this.audioSubmissionTime = Date.now() / 1000;
                    try {
                        this.metaInfo = this.metaInfo ?? (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                        if (!this.currentTurnStartTime) {
                            this.currentTurnStartTime = timestampMs();
                            this.currentTurnId = (this.metaInfo["turn_id"] ?? this.metaInfo["request_id"]) as string;
                        }
                    } catch { /* ignore */ }
                }

                // For HTTP non-streaming there's no real WS, just break on EOS
                const metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                if (metaInfo["eos"] === true) break;

                this.metaInfo = metaInfo;
                const startTime = timestampMs();
                const transcription = await this.getHttpTranscription(wsDataPacket["data"] as Buffer);
                const mi = transcription["meta_info"] as Record<string, unknown>;
                mi["include_latency"] = true;
                const elapsed = timestampMs() - startTime;
                mi["transcriber_first_result_latency"] = elapsed;
                mi["transcriber_total_stream_duration"] = elapsed;
                mi["transcriber_latency"] = elapsed;
                mi["audio_duration"] = mi["transcriber_duration"];
                mi["last_vocal_frame_timestamp"] = Date.now() / 1000;
                yield transcription;
            }
        } catch (e) {
            logger.info(`Cancelled sender task: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Sender stream (WebSocket)
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

                const data = wsDataPacket["data"];
                if (data) {
                    try {
                        ws.send(data as Buffer);
                    } catch (e) {
                        logger.error(`Error sending data to websocket: ${e}`);
                        break;
                    }
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
        // Wire WS events into promise-resolver queue
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

                    if (this.connectionStartTime === null) {
                        this.connectionStartTime = Date.now() / 1000 - this.numFrames * this.audioFrameDuration;
                    }

                    const msgType = msg["type"] as string;

                    // --- SpeechStarted ---
                    if (msgType === "SpeechStarted") {
                        logger.info("Received SpeechStarted from deepgram");
                        this.turnCounter++;
                        this.currentTurnId = this.turnCounter;
                        this.speechStartTime = timestampMs();
                        this.currentTurnInterimDetails = [];
                        logger.info(`Starting new turn: ${this.currentTurnId}`);
                        yield createWsDataPacket({ data: "speech_started", metaInfo: this.metaInfo! });

                        // --- Results ---
                    } else if (msgType === "Results") {
                        const channel = msg["channel"] as Record<string, unknown>;
                        const alternatives = (channel["alternatives"] as Record<string, unknown>[])[0]!;
                        const transcript = ((alternatives["transcript"] as string) ?? "").trim();
                        const deepgramRequestId = (msg["metadata"] as Record<string, unknown>)?.["request_id"];

                        if (transcript) {
                            this.setTranscriptionCursor(msg);
                            const audioPositionEnd = this.transcriptionCursor;
                            let latencyMs: number | null = null;
                            const audioSentAt = this.findAudioSendTimestamp(audioPositionEnd);
                            if (audioSentAt !== null) {
                                latencyMs = Math.round((timestampMs() - audioSentAt) * 100000) / 100000;
                            }

                            this.currentTurnInterimDetails.push({
                                transcript,
                                latency_ms: latencyMs,
                                is_final: msg["is_final"] ?? false,
                                received_at: Date.now() / 1000,
                                request_id: deepgramRequestId,
                            });

                            this.lastInterimTime = Date.now() / 1000;
                            logger.info(`Interim result - is_final: ${msg["is_final"]}, transcript: ${transcript}`);
                            yield createWsDataPacket({
                                data: { type: "interim_transcript_received", content: transcript },
                                metaInfo: this.metaInfo!,
                            });
                        }

                        if (msg["is_final"] && transcript) {
                            logger.info(`is_final=true: ${transcript}`);
                            this.finalTranscript += ` ${transcript}`;
                            if (this.isTranscriptSentForProcessing) this.isTranscriptSentForProcessing = false;
                        }

                        if (msg["speech_final"] && this.finalTranscript.trim()) {
                            if (!this.isTranscriptSentForProcessing) {
                                const finalText = this.finalTranscript.trim();
                                logger.info(`speech_final: yielding transcript: ${finalText}`);
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
                                    this.isTranscriptSentForProcessing = true;
                                } catch (e) {
                                    logger.error(`Failed to extract transcript in speech_final: ${e}`);
                                }
                                yield createWsDataPacket({
                                    data: { type: "transcript", content: finalText },
                                    metaInfo: this.metaInfo!,
                                });
                            }
                        }

                        // --- UtteranceEnd ---
                    } else if (msgType === "UtteranceEnd") {
                        logger.info(`UtteranceEnd: is_transcript_sent=${this.isTranscriptSentForProcessing}`);

                        if (!this.isTranscriptSentForProcessing && this.finalTranscript.trim()) {
                            const finalText = this.finalTranscript.trim();
                            logger.info(`UtteranceEnd: yielding transcript: ${finalText}`);

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
                                this.isTranscriptSentForProcessing = true;
                            } catch (e) {
                                logger.error(`Failed to extract transcript in UtteranceEnd: ${e}`);
                            }

                            yield createWsDataPacket({
                                data: { type: "transcript", content: finalText },
                                metaInfo: this.metaInfo!,
                            });
                        } else {
                            logger.info("UtteranceEnd: transcript already processed, yielding speech_ended");
                            this.speechStartTime = null;
                            this.speechEndTime = null;
                            this.currentTurnInterimDetails = [];
                            this.currentTurnStartTime = null;
                            this.currentTurnId = null;
                            this.finalTranscript = "";
                            yield createWsDataPacket({ data: { type: "speech_ended" }, metaInfo: this.metaInfo! });
                        }

                        // --- Metadata ---
                    } else if (msgType === "Metadata") {
                        const dur = msg["duration"] as number | undefined;
                        if (dur !== undefined) {
                            this.metaInfo!["deepgram_duration"] = dur;
                            logger.info(`Deepgram Metadata: duration=${dur}s`);
                        }
                    }
                } catch (e) {
                    console.error(e);
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

    getMetaInfo(): Record<string, unknown> {
        return this.metaInfo ?? {};
    }

    // ------------------------------------------------------------------
    // Connect
    // ------------------------------------------------------------------

    private async deepgramConnect(): Promise<WebSocket> {
        const wsUrl = this.getDeepgramWsUrl();
        logger.info(`Attempting to connect to Deepgram websocket: ${wsUrl}`);

        return new Promise<WebSocket>((resolve, reject) => {
            const timer = setTimeout(
                () => reject(new ConnectionError("Timeout connecting to Deepgram")),
                10_000
            );

            const ws = new WebSocket(wsUrl, {
                headers: { Authorization: `Token ${this.apiKey}` },
            });

            ws.once("open", () => {
                clearTimeout(timer);
                this.websocketConnection = ws;
                this.connectionAuthenticated = true;
                logger.info("Successfully connected to Deepgram websocket");
                resolve(ws);
            });

            ws.once("error", (err) => {
                clearTimeout(timer);
                reject(new ConnectionError(`Deepgram connection error: ${err.message}`));
            });
        });
    }

    // ------------------------------------------------------------------
    // Transcribe (main entry)
    // ------------------------------------------------------------------

    private async transcribe(): Promise<void> {
        let deepgramWs: WebSocket | null = null;

        try {
            const start = timestampMs();

            try {
                deepgramWs = await this.deepgramConnect();
            } catch (e) {
                logger.error(`Failed to establish Deepgram connection: ${e}`);
                await this.toggleConnection();
                return;
            }

            if (!this.connectionTime) {
                this.connectionTime = Math.round(timestampMs() - start);
            }

            if (this.stream) {
                this.senderTask = this.senderStream(deepgramWs);
                this.heartbeatTask = this.sendHeartbeat(deepgramWs);
                this.utteranceTimeoutTask = this.monitorUtteranceTimeout();

                try {
                    for await (const message of this.receiver(deepgramWs)) {
                        if (this.connectionOn) {
                            await this.pushToTranscriberQueue(message);
                        } else {
                            logger.info("Closing deepgram connection, waiting for Metadata");
                            await this._close(
                                { send: (d) => Promise.resolve(deepgramWs!.send(d)) },
                                { type: "CloseStream" }
                            );
                            // Wait up to 5s for Metadata
                            const deadline = Date.now() + 5_000;
                            for await (const _ of this.receiver(deepgramWs)) {
                                if (this.metaInfo?.["deepgram_duration"] !== undefined) break;
                                if (Date.now() > deadline) {
                                    logger.warn("Timeout waiting for Deepgram Metadata");
                                    break;
                                }
                            }
                            break;
                        }
                    }
                } catch (e) {
                    logger.error(`Error during streaming: ${e}`);
                    throw e;
                }

                await Promise.allSettled([
                    this.senderTask,
                    this.heartbeatTask,
                    this.utteranceTimeoutTask,
                ]);
            } else {
                for await (const message of this.sender()) {
                    await this.pushToTranscriberQueue(message);
                }
            }
        } catch (e) {
            logger.error(`Unexpected error in transcribe: ${e}`);
            await this.toggleConnection();
        } finally {
            this.utteranceAbort.abort();

            if (deepgramWs) {
                try {
                    deepgramWs.close();
                    logger.info("Deepgram websocket closed in finally block");
                } catch (e) {
                    logger.error(`Error closing websocket in finally: ${e}`);
                }
                this.websocketConnection = null;
                this.connectionAuthenticated = false;
            }

            if (this.metaInfo?.["deepgram_duration"] !== undefined) {
                this.metaInfo["transcriber_duration"] = this.metaInfo["deepgram_duration"];
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
            logger.error(`Error starting transcription task: ${e}`);
        }
    }
}

class ConnectionError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "ConnectionError";
    }
}