import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket, getSynthAudioFormat, resample, wavBytesToPcm, type WsDataPacket } from "../helper/utils";
import { SARVAM_MODEL_SAMPLING_RATE_MAPPING } from "../constants";
import { BaseSynthesizer } from "./base";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("sarvamSynthesizer");

export class SarvamSynthesizer extends BaseSynthesizer {
    private apiKey: string;
    private voiceId: string;
    private model: string;
    private samplingRate: number;
    private originalSamplingRate: number | null;
    private apiUrl: string;
    private wsUrl: string;
    private language: string;
    private loudness = 1.0;
    private pitch = 0.0;
    private pace: number;
    private enablePreprocessing = true;

    private firstChunkGenerated = false;
    private lastTextSent = false;
    private metaInfo: Record<string, unknown> | null = null;
    private synthesizedCharacters = 0;
    private websocketHolder: { websocket: WebSocket | null } = { websocket: null };
    private senderTask: Promise<void> | null = null;
    private conversationEnded = false;
    private currentTurnStartTime: number | null = null;
    private currentTurnId: string | number | null = null;
    private textQueue: Record<string, unknown>[] = [];
    private currentText = "";

    // Internal async generator plumbing for receiver
    private audioChunkResolvers: ((chunk: Buffer | null) => void)[] = [];
    private pendingChunks: (Buffer | null)[] = [];

    constructor(opts: {
        voiceId: string;
        model: string;
        language: string;
        samplingRate?: string;
        stream?: boolean;
        bufferSize?: number;
        speed?: number;
        synthesizerKey?: string | null;
        taskManagerInstance?: Record<string, unknown> | null;
        [key: string]: unknown;
    }) {
        super({
            taskManagerInstance: opts.taskManagerInstance ?? null,
            stream: opts.stream ?? false,
            bufferSize: opts.bufferSize ?? 400,
        });

        this.apiKey = opts.synthesizerKey ?? process.env.SARVAM_API_KEY!;
        this.voiceId = opts.voiceId;
        this.model = opts.model;
        this.stream = opts.stream ?? false;
        this.bufferSize = opts.bufferSize ?? 400;
        if (this.bufferSize < 30 || this.bufferSize > 200) this.bufferSize = 200;

        this.samplingRate = parseInt(opts.samplingRate ?? "8000", 10);
        this.originalSamplingRate = SARVAM_MODEL_SAMPLING_RATE_MAPPING[opts.model] ?? null;
        this.apiUrl = "https://api.sarvam.ai/text-to-speech";
        this.wsUrl = `wss://api.sarvam.ai/text-to-speech/ws?model=${opts.model}`;
        this.language = opts.language;
        this.pace = opts.speed ?? 1.0;
    }

    override getEngine(): string {
        return this.model;
    }

    override supportsWebsocket(): boolean {
        return true;
    }

    override getSleepTime(): number {
        return 0.01;
    }

    override getSynthesizedCharacters(): number {
        return this.synthesizedCharacters;
    }

    // ------------------------------------------------------------------
    // HTTP synthesis
    // ------------------------------------------------------------------

    private async sendPayload(payload: Record<string, unknown>): Promise<string | null> {
        const res = await fetch(this.apiUrl, {
            method: "POST",
            headers: {
                "api-subscription-key": this.apiKey,
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        if (res.ok) {
            const data = (await res.json()) as Record<string, unknown>;
            const audios = data["audios"] as string[] | undefined;
            if (Array.isArray(audios) && audios.length) return audios[0]!;
        } else {
            logger.error(`Error: ${res.status} - ${await res.text()}`);
        }
        return null;
    }

    private buildPayload(text: string): Record<string, unknown> {
        const payload: Record<string, unknown> = {
            target_language_code: this.language,
            text,
            speaker: this.voiceId,
            pitch: this.pitch,
            loudness: this.loudness,
            speech_sample_rate: this.samplingRate,
            enable_preprocessing: this.enablePreprocessing,
            model: this.model,
        };
        if (this.model === "bulbul:v3") {
            delete payload["pitch"];
            delete payload["loudness"];
        }
        return payload;
    }

    override async synthesize(text: string): Promise<string | null> {
        return this.sendPayload(this.buildPayload(text));
    }

    // ------------------------------------------------------------------
    // WebSocket connection
    // ------------------------------------------------------------------

    async establishConnection(): Promise<WebSocket | null> {
        const start = performance.now();
        return new Promise<WebSocket | null>((resolve) => {
            const timeout = setTimeout(() => {
                logger.error("Timeout while connecting to Sarvam TTS websocket");
                resolve(null);
            }, 10_000);

            const ws = new WebSocket(this.wsUrl, {
                headers: { "api-subscription-key": this.apiKey },
            });

            ws.once("open", async () => {
                clearTimeout(timeout);
                if (!this.connectionTime) {
                    this.connectionTime = Math.round(performance.now() - start);
                }
                const bosMessage = {
                    type: "config",
                    data: {
                        target_language_code: this.language,
                        speaker: this.voiceId,
                        pitch: this.pitch,
                        pace: this.pace,
                        loudness: this.loudness,
                        enable_preprocessing: this.enablePreprocessing,
                        output_audio_codec: "wav",
                        output_audio_bitrate: "32k",
                        max_chunk_length: 250,
                        min_buffer_size: this.bufferSize,
                    },
                };
                ws.send(JSON.stringify(bosMessage));
                logger.info(`Connected to ${this.wsUrl}`);

                // Wire up receiver piping
                ws.on("message", (raw) => {
                    try {
                        const data = JSON.parse(raw.toString()) as Record<string, unknown>;
                        if (data["type"] === "audio") {
                            const audio = (data["data"] as Record<string, unknown>)["audio"] as string;
                            this.pushAudioChunk(Buffer.from(audio, "base64"));
                        }
                    } catch (e) {
                        logger.error(`Error parsing Sarvam message: ${e}`);
                    }
                });

                ws.once("close", () => this.pushAudioChunk(null));
                ws.once("error", () => this.pushAudioChunk(null));

                resolve(ws);
            });

            ws.once("error", (err) => {
                clearTimeout(timeout);
                const msg = err.message;
                if (msg.includes("401") || msg.includes("403")) {
                    logger.error(`Sarvam TTS authentication failed: ${err}`);
                } else if (msg.includes("404")) {
                    logger.error(`Sarvam TTS endpoint not found: ${err}`);
                } else {
                    logger.error(`Sarvam TTS connection failed: ${err}`);
                }
                resolve(null);
            });
        });
    }

    // Push chunks into the async generator queue
    private pushAudioChunk(chunk: Buffer | null): void {
        const resolver = this.audioChunkResolvers.shift();
        if (resolver) {
            resolver(chunk);
        } else {
            this.pendingChunks.push(chunk);
        }
    }

    private async *receiver(): AsyncGenerator<Buffer> {
        while (true) {
            if (this.conversationEnded) return;

            const ws = this.websocketHolder.websocket;
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                await new Promise((r) => setTimeout(r, 100));
                continue;
            }

            const chunk = await new Promise<Buffer | null>((resolve) => {
                const pending = this.pendingChunks.shift();
                if (pending !== undefined) {
                    resolve(pending);
                } else {
                    this.audioChunkResolvers.push(resolve);
                }
            });

            if (chunk === null) break;
            yield chunk;

            if (this.lastTextSent) {
                yield Buffer.from([0x00]);
            }
        }
    }

    // ------------------------------------------------------------------
    // Monitor connection
    // ------------------------------------------------------------------

    override async monitorConnection(): Promise<void> {
        let consecutiveFailures = 0;
        const maxFailures = 3;

        while (consecutiveFailures < maxFailures) {
            const ws = this.websocketHolder.websocket;
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                logger.info("Re-establishing sarvam connection...");
                const result = await this.establishConnection();
                if (!result) {
                    consecutiveFailures++;
                    logger.warn(`Sarvam TTS connection failed (attempt ${consecutiveFailures}/${maxFailures})`);
                    if (consecutiveFailures >= maxFailures) {
                        logger.error("Max connection failures reached for Sarvam TTS");
                        break;
                    }
                } else {
                    this.websocketHolder.websocket = result;
                    consecutiveFailures = 0;
                }
            }
            await new Promise((r) => setTimeout(r, 1_000));
        }
    }

    // ------------------------------------------------------------------
    // Sender
    // ------------------------------------------------------------------

    async sender(text: string, sequenceId: number, endOfLlmStream = false): Promise<void> {
        try {
            if (this.conversationEnded) return;
            if (!this.shouldSynthesizeResponse(sequenceId)) {
                logger.info(`Not synthesizing: sequence_id ${sequenceId} not in current ids`);
                return;
            }

            // Wait for WS to be ready
            while (!this.websocketHolder.websocket || this.websocketHolder.websocket.readyState !== WebSocket.OPEN) {
                logger.info("Waiting for sarvam ws connection...");
                await new Promise((r) => setTimeout(r, 1_000));
            }

            if (text) {
                try {
                    this.websocketHolder.websocket.send(JSON.stringify({ type: "text", data: { text } }));
                } catch (e) {
                    logger.error(`Error sending chunk: ${e}`);
                    return;
                }
            }

            if (endOfLlmStream) this.lastTextSent = true;

            try {
                this.websocketHolder.websocket.send(JSON.stringify({ type: "flush" }));
            } catch (e) {
                logger.info(`Error sending flush: ${e}`);
            }
        } catch (e) {
            logger.error(`Unexpected error in sender: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Audio processing
    // ------------------------------------------------------------------

    private async processAudioData(audio: Buffer): Promise<Buffer | null> {
        const format = getSynthAudioFormat(audio);

        if (format === "wav" && this.model === "bulbul:v3") {
            const receivedSamplingRate = audio.readUInt32LE(24);
            if (this.originalSamplingRate !== receivedSamplingRate) {
                logger.warn(
                    `Expected sampling rate ${this.originalSamplingRate} but got ${receivedSamplingRate}, updating.`
                );
                this.originalSamplingRate = receivedSamplingRate;
            }
            return null; // Header only; subsequent chunks are PCM
        }

        try {
            const resampled = await resample(audio, this.samplingRate, {
                format,
                originalSampleRate: this.originalSamplingRate ?? undefined,
            });
            return format === "wav" ? wavBytesToPcm(resampled) : resampled;
        } catch (e) {
            logger.error(`Error resampling audio: ${e}`);
            return null;
        }
    }

    // ------------------------------------------------------------------
    // Generate
    // ------------------------------------------------------------------

    override  async *generate(): AsyncGenerator<WsDataPacket> {
        try {
            if (!this.stream) return;

            for await (const message of this.receiver()) {
                logger.info("Received message from server");

                if (this.textQueue.length) {
                    this.metaInfo = this.textQueue.shift()!;
                    try {
                        if (this.currentTurnStartTime !== null) {
                            const firstResultLatency = (performance.now() - this.currentTurnStartTime) / 1000;
                            this.metaInfo["synthesizer_latency"] = firstResultLatency;
                        }
                    } catch { /* ignore */ }
                }

                this.metaInfo!["format"] = "wav";

                if (!this.firstChunkGenerated) {
                    this.metaInfo!["is_first_chunk"] = true;
                    this.firstChunkGenerated = true;
                } else {
                    this.metaInfo!["is_first_chunk"] = false;
                }

                if (this.lastTextSent) {
                    this.firstChunkGenerated = false;
                    this.lastTextSent = true;
                }

                let audio: Buffer | null = message;

                if (message.length === 1 && message[0] === 0x00) {
                    logger.info("received null byte - end of stream");
                    this.metaInfo!["end_of_synthesizer_stream"] = true;
                    this.firstChunkGenerated = false;
                    try {
                        if (this.currentTurnStartTime !== null) {
                            const totalDuration = (performance.now() - this.currentTurnStartTime) / 1000;
                            this.turnLatencies.push({
                                turn_id: this.currentTurnId,
                                sequence_id: this.currentTurnId,
                                first_result_latency_ms: Math.round((this.metaInfo!["synthesizer_latency"] as number ?? 0) * 1000),
                                total_stream_duration_ms: Math.round(totalDuration * 1000),
                            });
                            this.currentTurnStartTime = null;
                            this.currentTurnId = null;
                        }
                    } catch { /* ignore */ }
                } else {
                    audio = await this.processAudioData(message);
                    if (audio === null) continue;
                }

                this.metaInfo!["mark_id"] = uuidv4();
                yield createWsDataPacket({
                    data: audio,
                    meta_info: this.metaInfo ?? undefined
                });
            }
        } catch (e) {
            console.error(e);
            logger.info(`Error in sarvam generate: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Push
    // ------------------------------------------------------------------

    override push(text: string): void {
        void this.handlePush(text);
    }

    private async handlePush(text: string): Promise<void> {
        if (this.stream) {
            const message = text as unknown as Record<string, unknown>;

            const metaInfo = message["meta_info"] as Record<string, unknown>;
            const data = message["data"] as string;

            this.currentText = data;
            this.synthesizedCharacters += data?.length ?? 0;

            const endOfLlmStream = !!metaInfo["end_of_llm_stream"];
            this.metaInfo = { ...metaInfo };
            metaInfo["text"] = data;

            try {
                this.currentTurnStartTime = performance.now();
                this.currentTurnId =
                    (metaInfo["turn_id"] ?? metaInfo["sequence_id"]) as string | number;
            } catch { }

            this.senderTask = this.sender(
                data,
                metaInfo["sequence_id"] as number,
                endOfLlmStream
            );

            this.textQueue.push({ ...metaInfo });

        } else {
            this.internalQueue.put_nowait({ data: text });
        }
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------

    override async cleanup(): Promise<void> {
        this.conversationEnded = true;
        logger.info("cleaning sarvam synthesizer tasks");

        if (this.senderTask) {
            try {
                await this.senderTask;
            } catch { /* ignore */ }
            this.senderTask = null;
        }

        if (this.websocketHolder.websocket) {
            this.websocketHolder.websocket.close();
            this.websocketHolder.websocket = null;
        }

        logger.info("WebSocket connection closed.");
    }
}