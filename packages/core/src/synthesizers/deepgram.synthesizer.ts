import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { convertAudioToWav, createWsDataPacket, type WsDataPacket } from "../helper/utils";
import { InmemoryScalarCache } from "../memory/cache/inmemory.cache";
import { BaseSynthesizer } from "./base";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("deepgramSynthesizer");

const DEEPGRAM_HOST = process.env.DEEPGRAM_HOST ?? "api.deepgram.com";
const DEEPGRAM_TTS_URL = `https://${DEEPGRAM_HOST}/v1/speak`;
const DEEPGRAM_TTS_WS_URL = `wss://${DEEPGRAM_HOST}/v1/speak`;

export class DeepgramSynthesizer extends BaseSynthesizer {
    private voice: string;
    private voiceId: string;
    private sampleRate: string;
    private model: string;
    private firstChunkGenerated = false;
    private apiKey: string;
    private useMulaw: boolean;
    private format: string;
    private synthesizedCharacters = 0;
    private caching: boolean;
    private cache: InmemoryScalarCache | null = null;
    private wsUrl: string;
    private websocketHolder: { websocket: WebSocket | null } = { websocket: null };
    private textQueue: Record<string, unknown>[] = [];
    private metaInfo: Record<string, unknown> | null = null;
    private lastTextSent = false;
    private senderTask: Promise<void> | null = null;
    private conversationEnded = false;
    private currentTurnStartTime: number | null = null;
    private currentTurnId: string | number | null = null;
    private currentText = "";
    private wsSendTime: number | null = null;
    private currentTurnTtfb: number | null = null;

    // Receiver async generator plumbing
    private chunkResolvers: ((chunk: Buffer | null) => void)[] = [];
    private pendingChunks: (Buffer | null)[] = [];

    constructor(opts: {
        voiceId: string;
        voice: string;
        audioFormat?: string;
        samplingRate?: string;
        stream?: boolean;
        bufferSize?: number;
        caching?: boolean;
        model?: string;
        synthesizerKey?: string;
        transcriberKey?: string;
        useMulaw?: boolean;
        taskManagerInstance?: Record<string, unknown> | null;
        [key: string]: unknown;
    }) {
        super({
            taskManagerInstance: opts.taskManagerInstance ?? null,
            stream: opts.stream ?? false,
            bufferSize: opts.bufferSize ?? 400,
        });

        this.voice = opts.voice;
        this.voiceId = opts.voiceId;
        this.sampleRate = opts.samplingRate ?? "8000";
        this.model = opts.model ?? "aura-zeus-en";
        this.apiKey = opts.transcriberKey ?? opts.synthesizerKey ?? process.env.DEEPGRAM_AUTH_TOKEN!;

        this.useMulaw = opts.useMulaw ?? false;
        const audioFormat = opts.audioFormat ?? "pcm";
        this.format = (this.useMulaw || ["pcm", "wav"].includes(audioFormat)) ? "mulaw" : audioFormat;

        // Append voice_id if model has only one dash segment
        if (this.model.split("-").length === 2) {
            this.model = `${this.model}-${this.voiceId}`;
        }

        this.caching = opts.caching ?? true;
        if (this.caching) this.cache = new InmemoryScalarCache();

        this.stream = opts.stream ?? false;
        this.wsUrl = `${DEEPGRAM_TTS_WS_URL}?encoding=${this.format}&sample_rate=${this.sampleRate}&model=${this.model}`;
    }

    override getSynthesizedCharacters(): number { return this.synthesizedCharacters; }
    override getEngine(): string { return this.model; }
    override supportsWebsocket(): boolean { return true; }
    override getSleepTime(): number { return this.stream ? 0.01 : super.getSleepTime(); }
    async openConnection(): Promise<void> { }

    // ------------------------------------------------------------------
    // Interruption
    // ------------------------------------------------------------------

    override async handleInterruption(): Promise<void> {
        try {
            const ws = this.websocketHolder.websocket;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "Clear" }));
                logger.info("Sent Clear message to Deepgram TTS WebSocket");
            }
        } catch (e) {
            logger.error(`Error handling interruption: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // HTTP synthesis
    // ------------------------------------------------------------------

    private async generateHttp(text: string): Promise<Buffer> {
        const url = `${DEEPGRAM_TTS_URL}?container=none&encoding=${this.format}&sample_rate=${this.sampleRate}&model=${this.model}`;
        logger.info(`Sending deepgram request ${url}`);

        try {
            const res = await fetch(url, {
                method: "POST",
                headers: {
                    Authorization: `Token ${this.apiKey}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text }),
            });

            if (res.ok) {
                const data = Buffer.from(await res.arrayBuffer());
                logger.info(`Deepgram response status ${res.status} length ${data.length}`);
                return data;
            } else {
                logger.info(`Deepgram error status ${res.status}`);
                return Buffer.from([0x00]);
            }
        } catch (e) {
            logger.error(`Deepgram HTTP error: ${e}`);
            return Buffer.from([0x00]);
        }
    }

    override async synthesize(text: string): Promise<Buffer> {
        try {
            const audio = await this.generateHttp(text);
            if (this.format === "mp3") return await convertAudioToWav(audio, "mp3");
            return audio;
        } catch (e) {
            logger.error(`Could not synthesize: ${e}`);
            return Buffer.from([0x00]);
        }
    }

    // ------------------------------------------------------------------
    // WebSocket connection
    // ------------------------------------------------------------------

    async establishConnection(): Promise<WebSocket | null> {
        const start = performance.now();

        return new Promise<WebSocket | null>((resolve) => {
            const timeout = setTimeout(() => {
                logger.error("Timeout while connecting to Deepgram TTS WebSocket");
                resolve(null);
            }, 10_000);

            const ws = new WebSocket(this.wsUrl, {
                headers: { Authorization: `Token ${this.apiKey}` },
            });

            ws.once("open", () => {
                clearTimeout(timeout);
                if (!this.connectionTime) {
                    this.connectionTime = Math.round(performance.now() - start);
                }
                logger.info(`Connected to Deepgram TTS WebSocket: ${this.wsUrl}`);

                // Wire receiver events
                ws.on("message", (raw, isBinary) => {
                    if (isBinary) {
                        // Binary = raw audio chunk
                        const chunk = Buffer.isBuffer(raw) ? raw : Buffer.from(raw as ArrayBuffer);
                        this.pushChunk(chunk);
                    } else {
                        try {
                            const data = JSON.parse(raw.toString()) as Record<string, unknown>;
                            const msgType = data["type"] as string;

                            if (msgType === "Metadata") {
                                logger.info(`Deepgram TTS Metadata: request_id=${data["request_id"]}`);
                            } else if (msgType === "Flushed") {
                                logger.info(`Deepgram TTS Flushed: sequence_id=${data["sequence_id"]}`);
                                this.pushChunk(Buffer.from([0x00])); // End-of-stream sentinel
                            } else if (msgType === "Cleared") {
                                logger.info(`Deepgram TTS Cleared: sequence_id=${data["sequence_id"]}`);
                            } else if (msgType === "Warning") {
                                logger.info(`Deepgram TTS Warning: ${data["description"]}`);
                            } else {
                                logger.info(`Deepgram TTS response: ${JSON.stringify(data)}`);
                            }
                        } catch {
                            logger.warn("Received unexpected non-JSON text from Deepgram");
                        }
                    }
                });

                ws.once("close", () => this.pushChunk(null));
                ws.once("error", () => this.pushChunk(null));

                resolve(ws);
            });

            ws.once("error", (err) => {
                clearTimeout(timeout);
                const msg = err.message;
                if (msg.includes("401")) {
                    logger.error("Deepgram authentication failed: Invalid API key");
                } else if (msg.includes("403")) {
                    logger.error("Deepgram authentication failed: Access forbidden");
                } else {
                    logger.error(`Deepgram WebSocket connection failed: ${err}`);
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
                logger.info("Re-establishing Deepgram TTS connection...");
                const result = await this.establishConnection();
                if (!result) {
                    consecutiveFailures++;
                    logger.warn(`Deepgram TTS connection failed (attempt ${consecutiveFailures}/${maxFailures})`);
                    if (consecutiveFailures >= maxFailures) {
                        logger.error("Max connection failures reached for Deepgram TTS");
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
                await this.flushSynthesizerStream();
                return;
            }

            // Wait for WS
            const wsWaitStart = performance.now();
            while (!this.websocketHolder.websocket || this.websocketHolder.websocket.readyState !== WebSocket.OPEN) {
                logger.info("Waiting for Deepgram TTS WebSocket...");
                await new Promise((r) => setTimeout(r, 500));
            }
            const wsWaitMs = performance.now() - wsWaitStart;
            if (wsWaitMs > 10) logger.info(`Deepgram sender ws_wait=${wsWaitMs.toFixed(0)}ms`);

            if (text) {
                if (!this.shouldSynthesizeResponse(sequenceId)) {
                    await this.flushSynthesizerStream();
                    return;
                }
                try {
                    if (this.wsSendTime === null) {
                        this.wsSendTime = performance.now();
                        logger.info("Deepgram WS send first_text_sent");
                    }
                    this.websocketHolder.websocket.send(JSON.stringify({ type: "Speak", text }));
                } catch (e) {
                    logger.error(`Error sending chunk to Deepgram: ${e}`);
                    return;
                }
            }

            if (endOfLlmStream) {
                this.lastTextSent = true;
                try {
                    this.websocketHolder.websocket.send(JSON.stringify({ type: "Flush" }));
                    logger.info("Sent Flush message to Deepgram TTS WebSocket");
                } catch (e) {
                    logger.error(`Error sending Flush: ${e}`);
                }
            }
        } catch (e) {
            logger.error(`Unexpected error in Deepgram sender: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Generate
    // ------------------------------------------------------------------

    override async *generate(): AsyncGenerator<WsDataPacket> {
        try {
            if (this.stream) {
                // WebSocket streaming mode
                for await (const message of this.receiver()) {
                    if (this.textQueue.length) {
                        this.metaInfo = this.textQueue.shift()!;
                        try {
                            if (this.currentTurnTtfb === null && this.wsSendTime !== null) {
                                this.currentTurnTtfb = (performance.now() - this.wsSendTime) / 1000;
                                this.metaInfo["synthesizer_latency"] = this.currentTurnTtfb;
                            }
                        } catch { /* ignore */ }
                    }

                    this.metaInfo!["format"] = this.useMulaw ? "mulaw" : this.format;
                    let audio: Buffer = message;

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
                        logger.info("Deepgram received null byte - end of stream");
                        this.metaInfo!["end_of_synthesizer_stream"] = true;
                        this.firstChunkGenerated = false;
                        try {
                            if (this.currentTurnStartTime !== null) {
                                const totalDuration = (performance.now() - this.currentTurnStartTime) / 1000;
                                this.turnLatencies.push({
                                    turn_id: this.currentTurnId,
                                    sequence_id: this.currentTurnId,
                                    first_result_latency_ms: Math.round((this.currentTurnTtfb ?? 0) * 1000),
                                    total_stream_duration_ms: Math.round(totalDuration * 1000),
                                });
                                this.currentTurnStartTime = null;
                                this.currentTurnId = null;
                                this.wsSendTime = null;
                                this.currentTurnTtfb = null;
                            }
                        } catch { /* ignore */ }
                    }

                    this.metaInfo!["mark_id"] = uuidv4();
                    yield createWsDataPacket({ data: audio, metaInfo: this.metaInfo! });
                }
            } else {
                // HTTP non-streaming mode
                while (true) {
                    const message = (await this.internalQueue.get()) as Record<string, unknown>;
                    logger.info(`Generating TTS response for message: ${JSON.stringify(message)}`);

                    const metaInfo = message["meta_info"] as Record<string, unknown>;
                    const text = message["data"] as string;

                    metaInfo["synthesizer_start_time"] = performance.now();

                    if (!this.shouldSynthesizeResponse(metaInfo["sequence_id"] as number)) {
                        logger.info(`Not synthesizing: sequence_id ${metaInfo["sequence_id"]} not in current ids`);
                        return;
                    }

                    let audioMessage: Buffer;

                    if (this.caching && this.cache) {
                        const cached = this.cache.get(text) as Buffer | null;
                        if (cached) {
                            logger.info(`Cache hit for: ${text}`);
                            audioMessage = cached;
                        } else {
                            logger.info("Cache miss");
                            this.synthesizedCharacters += text.length;
                            audioMessage = await this.generateHttp(text);
                            this.cache.set(text, audioMessage);
                        }
                    } else {
                        this.synthesizedCharacters += text.length;
                        audioMessage = await this.generateHttp(text);
                    }

                    if (this.format === "mp3") {
                        audioMessage = await convertAudioToWav(audioMessage, "mp3");
                    }

                    if (!this.firstChunkGenerated) {
                        metaInfo["is_first_chunk"] = true;
                        this.firstChunkGenerated = true;
                    } else {
                        metaInfo["is_first_chunk"] = false;
                    }

                    if (metaInfo["end_of_llm_stream"]) {
                        metaInfo["end_of_synthesizer_stream"] = true;
                        this.firstChunkGenerated = false;
                    }

                    metaInfo["text"] = text;

                    const startTime = metaInfo["synthesizer_start_time"] as number;
                    if (!("synthesizer_first_result_latency" in metaInfo)) {
                        const lat = (performance.now() - startTime) / 1000;
                        metaInfo["synthesizer_first_result_latency"] = lat;
                        metaInfo["synthesizer_latency"] = lat;
                    }

                    metaInfo["format"] = this.useMulaw ? "mulaw" : this.format;
                    metaInfo["text_synthesized"] = `${text} `;
                    metaInfo["mark_id"] = uuidv4();
                    metaInfo["synthesizer_total_stream_duration"] = (performance.now() - startTime) / 1000;

                    yield createWsDataPacket({ data: audioMessage, metaInfo });
                }
            }
        } catch (e) {
            console.error(e);
            logger.error(`Error in Deepgram generate: ${e}`);
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
        logger.info("Cleaning up Deepgram synthesizer tasks");

        if (this.senderTask) {
            try { await this.senderTask; } catch { /* ignore */ }
            this.senderTask = null;
        }

        const ws = this.websocketHolder.websocket;
        if (ws) {
            try {
                ws.send(JSON.stringify({ type: "Close" }));
                logger.info("Sent Close message to Deepgram TTS WebSocket");
            } catch (e) {
                logger.error(`Error sending Close message: ${e}`);
            }
            try { ws.close(); } catch (e) {
                logger.error(`Error closing WebSocket: ${e}`);
            }
        }

        this.websocketHolder.websocket = null;
        logger.info("Deepgram TTS WebSocket connection closed.");
    }
}