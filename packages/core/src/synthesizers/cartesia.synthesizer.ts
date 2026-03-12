import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { convertAudioToWav, createWsDataPacket, resample, type WsDataPacket } from "../helper/utils";
import { BaseSynthesizer } from "./base";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("cartesiaSynthesizer");

export class CartesiaSynthesizer extends BaseSynthesizer {
    private apiKey: string;
    private version = "2024-06-10";
    private language: string;
    private voiceId: string;
    private model: string;
    private samplingRate: string;
    private useMulaw = true;
    private firstChunkGenerated = false;
    private lastTextSent = false;
    private textQueue: Record<string, unknown>[] = [];
    private metaInfo: Record<string, unknown> | null = null;
    private synthesizedCharacters = 0;
    private websocketHolder: { websocket: WebSocket | null } = { websocket: null };
    private contextId: string | null = null;
    private wsRequestId: string | null = null;
    private senderTask: Promise<void> | null = null;
    private speed: number;
    private cartesiaHost: string;
    private wsUrl: string;
    private apiUrl: string;
    private turnId = 0;
    private sequenceId = 0;
    private contextIdsToIgnore = new Set<string>();
    private conversationEnded = false;
    private currentTurnStartTime: number | null = null;
    private currentTurnId: string | number | null = null;
    private currentText = "";

    // Async generator plumbing for receiver
    private chunkResolvers: ((chunk: Buffer | null) => void)[] = [];
    private pendingChunks: (Buffer | null)[] = [];

    constructor(opts: {
        voiceId: string;
        voice?: string;
        language?: string;
        model?: string;
        audioFormat?: string;
        samplingRate?: string;
        stream?: boolean;
        bufferSize?: number;
        synthesizerKey?: string | null;
        caching?: boolean;
        speed?: number;
        taskManagerInstance?: Record<string, unknown> | null;
        [key: string]: unknown;
    }) {
        super({
            taskManagerInstance: opts.taskManagerInstance ?? null,
            stream: opts.stream ?? false,
            bufferSize: opts.bufferSize ?? 400,
        });

        this.apiKey = opts.synthesizerKey ?? process.env.CARTESIA_API_KEY!;
        this.language = opts.language ?? "en";
        this.voiceId = opts.voiceId;
        this.model = opts.model ?? "sonic-english";
        this.stream = true; // Cartesia always streams
        this.samplingRate = opts.samplingRate ?? "16000";
        this.speed = opts.speed ?? 1.0;

        this.cartesiaHost = process.env.CARTESIA_API_HOST ?? "api.cartesia.ai";
        this.wsUrl = `wss://${this.cartesiaHost}/tts/websocket?api_key=${this.apiKey}&cartesia_version=${this.version}`;
        this.apiUrl = `https://${this.cartesiaHost}/tts/bytes`;
    }

    override getEngine(): string { return this.model; }
    override supportsWebsocket(): boolean { return true; }
    override getSleepTime(): number { return 0.01; }
    override getSynthesizedCharacters(): number { return this.synthesizedCharacters; }

    // ------------------------------------------------------------------
    // Interruption
    // ------------------------------------------------------------------

    override async handleInterruption(): Promise<void> {
        try {
            if (this.contextId) {
                this.contextIdsToIgnore.add(this.contextId);
                const interruptMessage = { context_id: this.contextId, cancel: true };
                logger.info(`handle_interruption: ${JSON.stringify(interruptMessage)}`);
                this.websocketHolder.websocket?.send(JSON.stringify(interruptMessage));
                this.contextId = null;
            }
        } catch (e) {
            logger.error(`Error in handle_interruption: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Payload builder
    // ------------------------------------------------------------------

    private formPayload(text: string): Record<string, unknown> {
        const payload: Record<string, unknown> = {
            context_id: this.contextId,
            model_id: this.model,
            transcript: text,
            language: this.language,
            voice: { mode: "id", id: this.voiceId },
            output_format: { container: "raw", encoding: "pcm_mulaw", sample_rate: 8000 },
            generation_config: { speed: this.speed },
        };
        if (text) payload["continue"] = true;
        return payload;
    }

    // ------------------------------------------------------------------
    // WebSocket connection
    // ------------------------------------------------------------------

    async establishConnection(): Promise<WebSocket | null> {
        const start = performance.now();

        return new Promise<WebSocket | null>((resolve) => {
            const timeout = setTimeout(() => {
                logger.error("Timeout while connecting to Cartesia websocket");
                resolve(null);
            }, 10_000);

            const ws = new WebSocket(this.wsUrl);

            ws.once("open", () => {
                clearTimeout(timeout);
                if (!this.connectionTime) {
                    this.connectionTime = Math.round(performance.now() - start);
                }
                // Wire receiver events
                ws.on("message", (raw) => {
                    try {
                        const data = JSON.parse(raw.toString()) as Record<string, unknown>;
                        const ctxId = data["context_id"] as string | undefined;

                        if (ctxId && this.contextIdsToIgnore.has(ctxId)) return;

                        if (data["data"]) {
                            const chunk = Buffer.from(data["data"] as string, "base64");
                            this.pushChunk(chunk);
                        } else if (data["done"]) {
                            logger.info(`Cartesia recv done context_id=${ctxId} request_id=${this.wsRequestId}`);
                            this.pushChunk(Buffer.from([0x00]));
                        } else {
                            logger.info(`No audio data context_id=${ctxId} request_id=${this.wsRequestId}`);
                        }
                    } catch (e) {
                        logger.error(`Error parsing Cartesia message: ${e}`);
                    }
                });

                ws.once("close", () => this.pushChunk(null));
                ws.once("error", () => this.pushChunk(null));

                logger.info(`Cartesia WebSocket connected connection_time=${this.connectionTime}ms`);
                resolve(ws);
            });

            ws.once("error", (err) => {
                clearTimeout(timeout);
                const msg = err.message;
                if (msg.includes("401") || msg.includes("403")) {
                    logger.error(`Cartesia authentication failed: ${err}`);
                } else if (msg.includes("404")) {
                    logger.error(`Cartesia endpoint not found: ${err}`);
                } else {
                    logger.error(`Cartesia handshake failed: ${err}`);
                }
                resolve(null);
            });
        });
    }

    // ------------------------------------------------------------------
    // Receiver async generator plumbing
    // ------------------------------------------------------------------

    private pushChunk(chunk: Buffer | null): void {
        const resolver = this.chunkResolvers.shift();
        if (resolver) resolver(chunk);
        else this.pendingChunks.push(chunk);
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
                if (pending !== undefined) resolve(pending);
                else this.chunkResolvers.push(resolve);
            });

            if (chunk === null) break;
            yield chunk;
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
                logger.info("Re-establishing cartesia connection...");
                const result = await this.establishConnection();
                if (!result) {
                    consecutiveFailures++;
                    logger.warn(`Cartesia connection failed (attempt ${consecutiveFailures}/${maxFailures})`);
                    if (consecutiveFailures >= maxFailures) {
                        logger.error("Max connection failures reached for Cartesia");
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

            while (!this.websocketHolder.websocket || this.websocketHolder.websocket.readyState !== WebSocket.OPEN) {
                logger.info("Waiting for Cartesia WebSocket...");
                await new Promise((r) => setTimeout(r, 1_000));
            }

            if (text) {
                try {
                    const payload = this.formPayload(text);
                    logger.info(`Cartesia sender context_id=${this.contextId} text_len=${text.length} request_id=${this.wsRequestId}`);
                    this.websocketHolder.websocket.send(JSON.stringify(payload));
                } catch (e) {
                    logger.error(`Error sending chunk context_id=${this.contextId}: ${e}`);
                    return;
                }
            }

            if (endOfLlmStream) {
                this.lastTextSent = true;
                logger.info(`Cartesia sender end_of_llm_stream context_id=${this.contextId}`);
                try {
                    this.websocketHolder.websocket.send(JSON.stringify(this.formPayload("")));
                } catch (e) {
                    logger.error(`Error sending end-of-stream: ${e}`);
                }
            }
        } catch (e) {
            logger.error(`Unexpected error in sender: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // HTTP synthesis
    // ------------------------------------------------------------------

    private async sendPayload(payload: Record<string, unknown>): Promise<Buffer | null> {
        const res = await fetch(this.apiUrl, {
            method: "POST",
            headers: {
                "X-API-Key": this.apiKey,
                "Cartesia-Version": this.version,
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });
        if (res.ok) return Buffer.from(await res.arrayBuffer());
        logger.error(`Error: ${res.status} - ${await res.text()}`);
        return null;
    }

    override async synthesize(text: string): Promise<Buffer | null> {
        return this.sendPayload({
            model_id: this.model,
            transcript: text,
            voice: { mode: "id", id: this.voiceId },
            output_format: { container: "mp3", encoding: "mp3", sample_rate: 44100 },
            language: this.language,
            generation_config: { speed: this.speed },
        });
    }

    // ------------------------------------------------------------------
    // Generate
    // ------------------------------------------------------------------

    override async *generate(): AsyncGenerator<WsDataPacket> {
        try {
            for await (const message of this.receiver()) {
                if (this.textQueue.length) {
                    this.metaInfo = this.textQueue.shift()!;
                    try {
                        if (this.currentTurnStartTime !== null) {
                            this.metaInfo["synthesizer_latency"] =
                                (performance.now() - this.currentTurnStartTime) / 1000;
                        }
                    } catch { /* ignore */ }
                }

                let audio: Buffer = message;

                if (this.useMulaw) {
                    this.metaInfo!["format"] = "mulaw";
                } else {
                    this.metaInfo!["format"] = "wav";
                    const wav = await convertAudioToWav(message, "mp3");
                    audio = await resample(wav, parseInt(this.samplingRate, 10), { format: "wav" });
                }

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
                                first_result_latency_ms: Math.round(
                                    ((this.metaInfo!["synthesizer_latency"] as number) ?? 0) * 1000
                                ),
                                total_stream_duration_ms: Math.round(totalDuration * 1000),
                            });
                            this.currentTurnStartTime = null;
                            this.currentTurnId = null;
                        }
                    } catch { /* ignore */ }
                }

                yield createWsDataPacket({ data: audio, metaInfo: this.metaInfo! });
            }
        } catch (e) {
            console.error(e);
            logger.error(`Error in cartesia generate: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Context management
    // ------------------------------------------------------------------

    private updateContext(metaInfo: Record<string, unknown>): void {
        this.contextId = uuidv4();
        this.turnId = (metaInfo["turn_id"] as number) ?? 0;
        this.sequenceId = (metaInfo["sequence_id"] as number) ?? 0;
        logger.info(
            `Cartesia new context_id=${this.contextId} turn_id=${this.turnId} sequence_id=${this.sequenceId}`
        );
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
        logger.info("cleaning cartesia synthesizer tasks");

        if (this.senderTask) {
            try { await this.senderTask; } catch { /* ignore */ }
            this.senderTask = null;
        }

        if (this.websocketHolder.websocket) {
            this.websocketHolder.websocket.close();
            this.websocketHolder.websocket = null;
        }
        logger.info("WebSocket connection closed.");
    }
}