import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket, ulaw2lin, timestampMs, type WsDataPacket } from "../helper/utils";
import { BaseTranscriber } from "./base";

const logger = configureLogger("assemblyAITranscriber");

export class AssemblyAITranscriber extends BaseTranscriber {
    private language: string;
    private stream: boolean;
    private provider: string;
    private heartbeatTask: Promise<void> | null = null;
    private senderTask: Promise<void> | null = null;
    private model: string;
    private samplingRate: number;
    private encoding: string;
    private formatTurns: boolean;
    private apiKey: string;
    private assemblyaiHost = "streaming.assemblyai.com";
    private transcriberOutputQueue: { put: (data: unknown) => Promise<void> } | null;
    private transcriptionTask: Promise<void> | null = null;

    private audioCursor = 0.0;
    private transcriptionCursor = 0.0;
    private interruptionSignalled = false;
    private apiUrl: string;

    private audioSubmitted = false;
    private audioSubmissionTime: number | null = null;
    private numFrames = 0;
    private connectionStartTime: number | null = null;
    private audioFrameDuration = 0.0;
    private audioFrameTimestamps: [number, number, number][] = [];
    private connectedViaDashboard: boolean;

    private sessionId: string | null = null;
    private currentTranscript = "";
    private isTranscriptSentForProcessing = false;
    private websocketConnection: WebSocket | null = null;
    private connectionAuthenticated = false;
    private currentTurnStartTime: number | null = null;
    private currentTurnId: string | null = null;
    private currentTurnInterimDetails: Record<string, unknown>[] = [];

    // Receiver plumbing
    private messageResolvers: ((msg: string | null) => void)[] = [];
    private pendingMessages: (string | null)[] = [];

    constructor(opts: {
        telephonyProvider: string;
        inputQueue?: { get: () => Promise<unknown> } | null;
        model?: string;
        stream?: boolean;
        language?: string;
        samplingRate?: string;
        encoding?: string;
        outputQueue?: { put: (data: unknown) => Promise<void> } | null;
        formatTurns?: boolean;
        transcriberKey?: string;
        enforceStreaming?: boolean;
        [key: string]: unknown;
    }) {
        super(opts.inputQueue ?? null);

        this.language = opts.language ?? "en";
        this.stream = opts.stream ?? true;
        this.provider = opts.telephonyProvider;
        this.model = opts.model ?? "universal-streaming";
        this.samplingRate = parseInt(opts.samplingRate ?? "16000", 10);
        this.encoding = opts.encoding ?? "pcm_s16le";
        this.formatTurns = opts.formatTurns ?? true;
        this.apiKey = (opts.transcriberKey as string | undefined) ?? process.env.ASSEMBLY_API_KEY!;
        this.transcriberOutputQueue = opts.outputQueue ?? null;
        this.connectedViaDashboard = (opts.enforceStreaming as boolean | undefined) ?? true;
        this.apiUrl = "https://api.assemblyai.com/v2/transcript";
        this.metaInfo = {};
    }

    // ------------------------------------------------------------------
    // URL builder
    // ------------------------------------------------------------------

    private getAssemblyAIWsUrl(): string {
        const params: Record<string, string | number | boolean> = {
            sample_rate: this.samplingRate,
            format_turns: this.formatTurns,
        };

        if (["twilio", "exotel", "plivo", "vobiz"].includes(this.provider)) {
            this.encoding = this.provider === "twilio" ? "mulaw" : "linear16";
            this.samplingRate = 8000;
            this.audioFrameDuration = 0.2;
            params["sample_rate"] = 8000;
        } else if (this.provider === "web_based_call") {
            this.encoding = "linear16";
            this.samplingRate = 16000;
            this.audioFrameDuration = 0.256;
            params["sample_rate"] = 16000;
        } else if (!this.connectedViaDashboard) {
            this.encoding = "linear16";
            params["sample_rate"] = 16000;
        }

        if (this.provider === "playground") {
            this.samplingRate = 8000;
            this.audioFrameDuration = 0.0;
        }

        if (this.language !== "en") {
            logger.warn("AssemblyAI Universal Streaming currently only supports English");
        }

        const qs = Object.entries(params).map(([k, v]) => `${k}=${v}`).join("&");
        return `wss://${this.assemblyaiHost}/v3/ws?${qs}`;
    }

    // ------------------------------------------------------------------
    // Connection
    // ------------------------------------------------------------------

    private async assemblyaiConnect(): Promise<WebSocket> {
        const wsUrl = this.getAssemblyAIWsUrl();
        logger.info(`Attempting to connect to AssemblyAI websocket: ${wsUrl}`);

        return new Promise<WebSocket>((resolve, reject) => {
            const timer = setTimeout(
                () => reject(new ConnectionError("Timeout while connecting to AssemblyAI websocket")),
                10_000
            );

            const ws = new WebSocket(wsUrl, { headers: { Authorization: this.apiKey } });

            ws.once("open", () => {
                clearTimeout(timer);
                this.websocketConnection = ws;
                this.connectionAuthenticated = true;
                logger.info("Successfully connected to AssemblyAI websocket");

                // Wire receiver
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

                resolve(ws);
            });

            ws.once("error", (err) => {
                clearTimeout(timer);
                reject(new ConnectionError(`AssemblyAI connection error: ${err.message}`));
            });
        });
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
                    ws.ping();
                } catch (e) {
                    logger.info(`Connection closed while sending ping: ${e}`);
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

        if (this.websocketConnection) {
            try {
                this.websocketConnection.send(JSON.stringify({ type: "Terminate" }));
                this.websocketConnection.close();
                logger.info("AssemblyAI WebSocket connection closed successfully");
            } catch (e) {
                logger.error(`Error closing websocket connection: ${e}`);
            } finally {
                this.websocketConnection = null;
                this.connectionAuthenticated = false;
            }
        }
    }

    override async cleanup(): Promise<void> {
        logger.info("Cleaning up AssemblyAI transcriber resources");

        if (this.websocketConnection) {
            try { this.websocketConnection.close(); } catch (e) {
                logger.error(`Error closing AssemblyAI websocket: ${e}`);
            }
            this.websocketConnection = null;
            this.connectionAuthenticated = false;
        }
        logger.info("AssemblyAI transcriber cleanup complete");
    }

    // ------------------------------------------------------------------
    // HTTP transcription (non-streaming)
    // ------------------------------------------------------------------

    private async getHttpTranscription(audioData: Buffer): Promise<WsDataPacket> {
        const headers: Record<string, string> = { Authorization: this.apiKey };
        this.currentRequestId = BaseTranscriber.generateRequestId();
        this.metaInfo!["request_id"] = this.currentRequestId;
        const startTime = Date.now() / 1000;

        // Upload
        const uploadRes = await fetch("https://api.assemblyai.com/v2/upload", {
            method: "POST",
            headers,
            body: Buffer.from(audioData).buffer,
        });
        if (!uploadRes.ok) throw new Error(`Failed to upload audio: ${uploadRes.status}`);
        const uploadJson = (await uploadRes.json()) as Record<string, unknown>;
        const audioUrl = uploadJson["upload_url"] as string;

        // Submit
        const submitRes = await fetch(this.apiUrl, {
            method: "POST",
            headers: { ...headers, "Content-Type": "application/json" },
            body: JSON.stringify({
                audio_url: audioUrl,
                language_code: this.language !== "en" ? this.language : undefined,
            }),
        });
        if (!submitRes.ok) throw new Error(`Failed to submit transcription: ${submitRes.status}`);
        const submitJson = (await submitRes.json()) as Record<string, unknown>;
        const transcriptId = submitJson["id"] as string;

        // Poll
        while (true) {
            const pollRes = await fetch(`${this.apiUrl}/${transcriptId}`, { headers });
            const result = (await pollRes.json()) as Record<string, unknown>;

            if (result["status"] === "completed") {
                const transcript = (result["text"] as string) ?? "";
                this.metaInfo!["start_time"] = startTime;
                this.metaInfo!["transcriber_latency"] = Date.now() / 1000 - startTime;
                this.metaInfo!["transcriber_duration"] = result["audio_duration"] ?? 0;
                return createWsDataPacket({ data: transcript, metaInfo: this.metaInfo! });
            } else if (result["status"] === "error") {
                throw new Error(`Transcription failed: ${result["error"]}`);
            }

            await new Promise((r) => setTimeout(r, 1_000));
        }
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
            ws.send(JSON.stringify({ type: "Terminate" }));
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Audio frame timestamp lookup
    // ------------------------------------------------------------------

    private findAudioSendTimestamp(audioPosition: number): number | null {
        for (const [frameStart, frameEnd, sendTs] of this.audioFrameTimestamps) {
            if (frameStart <= audioPosition && audioPosition <= frameEnd) return sendTs;
        }
        return null;
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
                }

                const metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                // For HTTP mode there's no real WS, so just break on EOS
                if (metaInfo["eos"] === true) break;

                if (wsDataPacket) {
                    this.metaInfo = metaInfo;
                    const startTime = performance.now();
                    const transcription = await this.getHttpTranscription(wsDataPacket["data"] as Buffer);
                    const elapsed = (performance.now() - startTime) / 1000;
                    const mi = transcription["meta_info"] as Record<string, unknown>;
                    mi["include_latency"] = true;
                    mi["transcriber_first_result_latency"] = elapsed;
                    mi["transcriber_total_stream_duration"] = elapsed;
                    mi["transcriber_latency"] = elapsed;
                    mi["audio_duration"] = mi["transcriber_duration"];
                    mi["last_vocal_frame_timestamp"] = Date.now() / 1000;
                    yield transcription;
                }
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

                if (!this.audioSubmitted && wsDataPacket) {
                    this.metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                    this.audioSubmitted = true;
                    this.audioSubmissionTime = Date.now() / 1000;
                    this.currentRequestId = BaseTranscriber.generateRequestId();
                    this.metaInfo["request_id"] = this.currentRequestId;
                    if (!this.currentTurnStartTime) {
                        this.currentTurnStartTime = performance.now();
                        this.currentTurnId = (this.metaInfo["turn_id"] ?? this.metaInfo["request_id"]) as string;
                    }
                }

                const eos = await this.checkAndProcessEndOfStream(wsDataPacket, ws);
                if (eos) break;

                const frameStart = this.numFrames * this.audioFrameDuration;
                const frameEnd = (this.numFrames + 1) * this.audioFrameDuration;
                const sendTimestamp = timestampMs();
                this.audioFrameTimestamps.push([frameStart, frameEnd, sendTimestamp]);
                this.numFrames++;
                this.audioCursor = this.numFrames * this.audioFrameDuration;

                let audioData = wsDataPacket["data"] as Buffer | undefined;
                if (Buffer.isBuffer(audioData)) {
                    try {
                        if (this.provider === "twilio" && this.encoding === "mulaw") {
                            audioData = ulaw2lin(audioData);
                        }
                        ws.send(audioData);
                    } catch (e) {
                        logger.error(`Error sending data to websocket: ${e}`);
                        break;
                    }
                } else if (audioData !== undefined) {
                    logger.warn(`Expected bytes for audio data, got: ${typeof audioData}`);
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

    private async *receiver(): AsyncGenerator<WsDataPacket> {
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

                    const messageType = msg["type"] as string;

                    // --- Begin ---
                    if (messageType === "Begin") {
                        this.sessionId = msg["id"] as string;
                        logger.info(`AssemblyAI session began. ID: ${this.sessionId}, expires: ${msg["expires_at"]}`);
                        yield createWsDataPacket({ data: "session_started", metaInfo: this.metaInfo! });

                        // --- Turn ---
                    } else if (messageType === "Turn") {
                        const transcript = ((msg["transcript"] as string) ?? "").trim();
                        const turnIsFormatted = !!(msg["turn_is_formatted"]);

                        if (transcript) {
                            let latencyMs: number | null = null;
                            const words = (msg["words"] as Record<string, unknown>[]) ?? [];

                            if (words.length) {
                                const lastWord = words[words.length - 1]!;
                                const audioPosSec = ((lastWord["end"] as number) ?? 0) / 1000;
                                const sentAt = this.findAudioSendTimestamp(audioPosSec);
                                if (sentAt !== null) {
                                    latencyMs = Math.round((timestampMs() - sentAt) * 100000) / 100000;
                                }
                            }

                            if (turnIsFormatted) {
                                logger.info(`Received formatted transcript: ${transcript}`);

                                if (this.currentTurnStartTime === null) {
                                    this.currentTurnStartTime = performance.now();
                                    this.currentTurnId = BaseTranscriber.generateRequestId();
                                    this.currentTurnInterimDetails = [];
                                    if (this.metaInfo && "transcriber_first_result_latency" in this.metaInfo) {
                                        delete this.metaInfo["transcriber_first_result_latency"];
                                    }
                                }

                                this.currentTurnInterimDetails.push({
                                    transcript, latency_ms: latencyMs, is_final: true, received_at: Date.now() / 1000,
                                });

                                try {
                                    if (this.currentTurnStartTime !== null) {
                                        const totalSec = (performance.now() - this.currentTurnStartTime) / 1000;
                                        this.metaInfo!["transcriber_total_stream_duration"] = totalSec;
                                        this.metaInfo!["transcriber_latency"] = totalSec;

                                        const [firstToFinal, lastToFinal] = this.calculateInterimToFinalLatencies(
                                            this.currentTurnInterimDetails as { received_at?: number }[]
                                        );

                                        this.turnLatencies.push({
                                            turn_id: this.currentTurnId,
                                            sequence_id: this.currentTurnId,
                                            first_result_latency_ms: Math.round(
                                                ((this.metaInfo?.["transcriber_first_result_latency"] as number) ?? 0) * 1000
                                            ),
                                            total_stream_duration_ms: Math.round(totalSec * 1000),
                                            interim_details: this.currentTurnInterimDetails,
                                            first_interim_to_final_ms: firstToFinal,
                                            last_interim_to_final_ms: lastToFinal,
                                        });

                                        this.currentTurnStartTime = null;
                                        this.currentTurnId = null;
                                        this.currentTurnInterimDetails = [];
                                    }
                                } catch (e) {
                                    logger.error(`Error tracking turn latencies: ${e}`);
                                }

                                yield createWsDataPacket({
                                    data: { type: "transcript", content: transcript },
                                    metaInfo: this.metaInfo!,
                                });

                            } else {
                                logger.debug(`Received interim transcript: ${transcript}`);

                                if (this.currentTurnStartTime === null) {
                                    this.currentTurnStartTime = performance.now();
                                    this.currentTurnId = BaseTranscriber.generateRequestId();
                                    this.currentTurnInterimDetails = [];
                                    if (this.metaInfo && "transcriber_first_result_latency" in this.metaInfo) {
                                        delete this.metaInfo["transcriber_first_result_latency"];
                                    }
                                }

                                this.currentTurnInterimDetails.push({
                                    transcript, latency_ms: latencyMs, is_final: false, received_at: Date.now() / 1000,
                                });

                                try {
                                    if (this.currentTurnStartTime !== null && this.metaInfo &&
                                        !("transcriber_first_result_latency" in this.metaInfo)) {
                                        const firstLat = (performance.now() - this.currentTurnStartTime) / 1000;
                                        this.metaInfo["transcriber_first_result_latency"] = firstLat;
                                        this.metaInfo["transcriber_latency"] = firstLat;
                                    }
                                } catch { /* ignore */ }

                                yield createWsDataPacket({
                                    data: { type: "interim_transcript_received", content: transcript },
                                    metaInfo: this.metaInfo!,
                                });
                            }
                        }

                        // --- Termination ---
                    } else if (messageType === "Termination") {
                        logger.info(`AssemblyAI session terminated. Audio: ${msg["audio_duration_seconds"]}s`);
                        yield createWsDataPacket({ data: "transcriber_connection_closed", metaInfo: this.metaInfo! });
                        return;

                        // --- Error ---
                    } else if (messageType === "Error") {
                        logger.error(`AssemblyAI error: ${msg["error"] ?? "Unknown error"}`);
                        yield createWsDataPacket({ data: "transcriber_error", metaInfo: this.metaInfo! });

                    } else {
                        logger.debug(`Received unknown message type: ${messageType}`);
                    }
                } catch (e) {
                    logger.error(`Error processing message: ${e}`);
                    console.error(e);
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
        if (this.transcriberOutputQueue) {
            await this.transcriberOutputQueue.put(dataPacket);
        }
    }

    getMetaInfo(): Record<string, unknown> {
        return this.metaInfo ?? {};
    }

    // ------------------------------------------------------------------
    // Transcribe (main entry)
    // ------------------------------------------------------------------

    private async transcribe(): Promise<void> {
        let assemblyaiWs: WebSocket | null = null;

        try {
            const start = performance.now();

            try {
                assemblyaiWs = await this.assemblyaiConnect();
            } catch (e) {
                logger.error(`Failed to establish AssemblyAI connection: ${e}`);
                await this.toggleConnection();
                return;
            }

            if (!this.connectionTime) {
                this.connectionTime = Math.round(performance.now() - start);
            }

            if (this.stream) {
                this.senderTask = this.senderStream(assemblyaiWs);
                this.heartbeatTask = this.sendHeartbeat(assemblyaiWs);

                try {
                    for await (const message of this.receiver()) {
                        if (this.connectionOn) {
                            await this.pushToTranscriberQueue(message);
                        } else {
                            logger.info("Closing the AssemblyAI connection");
                            assemblyaiWs.send(JSON.stringify({ type: "Terminate" }));
                            break;
                        }
                    }
                } catch (e) {
                    logger.error(`Error during streaming: ${e}`);
                    throw e;
                }

                await Promise.allSettled([this.senderTask, this.heartbeatTask]);
            } else {
                for await (const message of this.sender()) {
                    await this.pushToTranscriberQueue(message);
                }
            }
        } catch (e) {
            logger.error(`Unexpected error in transcribe: ${e}`);
            await this.toggleConnection();
        } finally {
            if (assemblyaiWs) {
                try {
                    assemblyaiWs.close();
                    logger.info("AssemblyAI websocket closed in finally block");
                } catch (e) {
                    logger.error(`Error closing websocket in finally block: ${e}`);
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