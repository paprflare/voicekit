import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { convertAudioToWav, createWsDataPacket, resample, type WsDataPacket } from "../helper/utils";
import { InmemoryScalarCache } from "../memory/cache/inmemory.cache";
import { BaseSynthesizer } from "./base";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("elevenlabsSynthesizer");

export class ElevenlabsSynthesizer extends BaseSynthesizer {
    private apiKey: string;
    private voice: string;
    private model: string;
    private samplingRate: string;
    private speed: number;
    private style: number;
    private audioFormat = "mp3";
    private useMulaw: boolean;
    private elevenlabsHost: string;
    private wsUrl: string;
    private apiUrl: string;
    private firstChunkGenerated = false;
    private lastTextSent = false;
    private textQueue: Record<string, unknown>[] = [];
    private metaInfo: Record<string, unknown> | null = null;
    private temperature: number;
    private similarityBoost: number;
    private caching: boolean;
    private cache: InmemoryScalarCache | null = null;
    private synthesizedCharacters = 0;
    private websocketHolder: { websocket: WebSocket | null } = { websocket: null };
    private senderTask: Promise<void> | null = null;
    private conversationEnded = false;
    private currentTurnStartTime: number | null = null;
    private currentTurnId: string | number | null = null;
    private currentText = "";
    private contextId: string | null = null;
    private wsSendTime: number | null = null;
    private wsTraceId: string | null = null;
    private currentTurnTtfb: number | null = null;

    // Receiver async generator plumbing — yields [Buffer, string] tuples
    private chunkResolvers: ((val: [Buffer, string] | null) => void)[] = [];
    private pendingChunks: ([Buffer, string] | null)[] = [];

    constructor(opts: {
        voice: string;
        voiceId: string;
        model?: string;
        audioFormat?: string;
        samplingRate?: string;
        stream?: boolean;
        bufferSize?: number;
        temperature?: number;
        similarityBoost?: number;
        speed?: number;
        style?: number;
        synthesizerKey?: string | null;
        caching?: boolean;
        useMulaw?: boolean;
        taskManagerInstance?: Record<string, unknown> | null;
        [key: string]: unknown;
    }) {
        super({
            taskManagerInstance: opts.taskManagerInstance ?? null,
            stream: opts.stream ?? false,
            bufferSize: opts.bufferSize ?? 400,
        });

        this.apiKey = opts.synthesizerKey ?? process.env.ELEVENLABS_API_KEY!;
        this.voice = opts.voiceId;
        this.model = opts.model ?? "eleven_turbo_v2_5";
        this.stream = true; // ElevenLabs always streams
        this.samplingRate = opts.samplingRate ?? "16000";
        this.speed = opts.speed ?? 1.0;
        this.style = opts.style ?? 0;
        this.temperature = opts.temperature ?? 0.5;
        this.similarityBoost = opts.similarityBoost ?? 0.75;
        this.useMulaw = opts.useMulaw ?? true;
        this.caching = opts.caching ?? true;
        if (this.caching) this.cache = new InmemoryScalarCache();

        this.elevenlabsHost = process.env.ELEVENLABS_API_HOST ?? "api.elevenlabs.io";
        const outputFormat = this.useMulaw ? "ulaw_8000" : "mp3_44100_128";
        this.wsUrl = `wss://${this.elevenlabsHost}/v1/text-to-speech/${this.voice}/multi-stream-input?model_id=${this.model}&output_format=${outputFormat}&inactivity_timeout=170&sync_alignment=true&optimize_streaming_latency=4`;
        this.apiUrl = `https://${this.elevenlabsHost}/v1/text-to-speech/${this.voice}?optimize_streaming_latency=2&output_format=`;
    }

    override getEngine(): string { return this.model; }
    override supportsWebsocket(): boolean { return true; }
    override getSynthesizedCharacters(): number { return this.synthesizedCharacters; }

    private getFormat(): string {
        return this.useMulaw ? "ulaw_8000" : "mp3_44100_128";
    }

    // ------------------------------------------------------------------
    // Interruption
    // ------------------------------------------------------------------

    override async handleInterruption(): Promise<void> {
        try {
            if (this.contextId) {
                const interruptMessage = { context_id: this.contextId, close_context: true };
                this.contextId = uuidv4();
                this.websocketHolder.websocket?.send(JSON.stringify(interruptMessage));
            }
        } catch { /* ignore */ }
    }

    // ------------------------------------------------------------------
    // WebSocket connection
    // ------------------------------------------------------------------

    async establishConnection(): Promise<WebSocket | null> {
        const start = performance.now();

        return new Promise<WebSocket | null>((resolve) => {
            const ws = new WebSocket(this.wsUrl);

            ws.once("open", () => {
                // Try to extract x-trace-id (not accessible from ws lib directly, best effort)
                this.wsTraceId = null;
                logger.info(`Elevenlabs WebSocket connected trace_id=${this.wsTraceId}`);

                const bosMessage = {
                    text: " ",
                    voice_settings: {
                        stability: this.temperature,
                        similarity_boost: this.similarityBoost,
                        speed: this.speed,
                        style: this.style,
                    },
                    generation_config: {
                        chunk_length_schedule: [50, 80, 120, 150],
                    },
                    xi_api_key: this.apiKey,
                };

                ws.send(JSON.stringify(bosMessage));

                if (!this.connectionTime) {
                    this.connectionTime = Math.round(performance.now() - start);
                }
                logger.info(`Connected to ${this.wsUrl}`);

                // Wire receiver events
                ws.on("message", (raw) => {
                    try {
                        const data = JSON.parse(raw.toString()) as Record<string, unknown>;
                        const recvStart = performance.now();

                        if (data["audio"] && this.wsSendTime !== null) {
                            // Latency logging handled in receiver via timing fields
                        }

                        logger.info(`response for isFinal: ${data["isFinal"] ?? false}`);

                        if (data["audio"]) {
                            const chunk = Buffer.from(data["audio"] as string, "base64");
                            let textSpoken = "";
                            try {
                                const chars = (data["alignment"] as Record<string, unknown>)?.["chars"] as string[] ?? [];
                                textSpoken = chars.join("");
                            } catch { /* ignore */ }
                            this.pushChunk([chunk, textSpoken]);
                        }

                        if (data["isFinal"]) {
                            logger.info(`WS recv isFinal trace_id=${this.wsTraceId}`);
                            this.pushChunk([Buffer.from([0x00]), ""]);
                        } else if (this.lastTextSent && !data["audio"]) {
                            try {
                                const chars = ((data["alignment"] as Record<string, unknown>)?.["chars"] as string[]) ?? [];
                                const responseText = chars.join("");
                                const lastFourWords = responseText.split(" ").slice(-4).join(" ").replace(/"/g, "").trim();
                                const currentNorm = this.normalizeText(this.currentText.trim()).replace(/"/g, "").trim();

                                logger.info(`Last four char - ${lastFourWords} | current text - ${currentNorm}`);

                                if (currentNorm.endsWith(lastFourWords)) {
                                    logger.info("send end_of_synthesizer_stream");
                                    this.pushChunk([Buffer.from([0x00]), ""]);
                                } else if (currentNorm.replace(/"/g, "").replace(/\s/g, "").endsWith(lastFourWords.replace(/\s/g, ""))) {
                                    logger.info("send end_of_synthesizer_stream on fallback");
                                    this.pushChunk([Buffer.from([0x00]), ""]);
                                }
                            } catch (e) {
                                logger.error(`Error getting chars from response: ${e}`);
                                this.pushChunk([Buffer.from([0x00]), ""]);
                            }
                        } else if (!data["audio"]) {
                            logger.info("No audio data in the response");
                        }
                    } catch (e) {
                        logger.error(`Error parsing ElevenLabs message: ${e}`);
                    }
                });

                ws.once("close", () => this.pushChunk(null));
                ws.once("error", () => this.pushChunk(null));

                resolve(ws);
            });

            ws.once("error", (err) => {
                logger.info(`Failed to connect: ${err}`);
                resolve(null);
            });
        });
    }

    // ------------------------------------------------------------------
    // Receiver async generator plumbing
    // ------------------------------------------------------------------

    private pushChunk(val: [Buffer, string] | null): void {
        const resolver = this.chunkResolvers.shift();
        if (resolver) resolver(val);
        else this.pendingChunks.push(val);
    }

    private async *receiver(): AsyncGenerator<[Buffer, string]> {
        while (true) {
            if (this.conversationEnded) return;

            const ws = this.websocketHolder.websocket;
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                await new Promise((r) => setTimeout(r, 100));
                continue;
            }

            const val = await new Promise<[Buffer, string] | null>((resolve) => {
                const pending = this.pendingChunks.shift();
                if (pending !== undefined) resolve(pending);
                else this.chunkResolvers.push(resolve);
            });

            if (val === null) break;
            yield val;
        }
    }

    // ------------------------------------------------------------------
    // Monitor connection
    // ------------------------------------------------------------------

    override async monitorConnection(): Promise<void> {
        while (true) {
            const ws = this.websocketHolder.websocket;
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                logger.info("Re-establishing elevenlabs connection...");
                this.websocketHolder.websocket = await this.establishConnection();
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
                logger.info("Waiting for elevenlabs ws connection...");
                await new Promise((r) => setTimeout(r, 1_000));
            }
            const wsWaitMs = performance.now() - wsWaitStart;
            if (wsWaitMs > 10) logger.info(`EL sender ws_wait=${wsWaitMs.toFixed(0)}ms trace_id=${this.wsTraceId}`);

            if (text) {
                for (const textChunk of this.textChunker(text)) {
                    if (!this.shouldSynthesizeResponse(sequenceId)) {
                        logger.info(`Not synthesizing (inner loop): sequence_id ${sequenceId}`);
                        await this.flushSynthesizerStream();
                        return;
                    }
                    try {
                        if (this.wsSendTime === null) {
                            this.wsSendTime = performance.now();
                            logger.info(`WS send trace_id=${this.wsTraceId} first_text_sent`);
                        }
                        this.websocketHolder.websocket.send(JSON.stringify({ text: textChunk }));
                    } catch (e) {
                        logger.info(`Error sending chunk: ${e}`);
                        return;
                    }
                }
            }

            if (endOfLlmStream) {
                this.lastTextSent = true;
                this.contextId = uuidv4();
            }

            try {
                this.websocketHolder.websocket.send(JSON.stringify({ text: "", flush: true }));
            } catch (e) {
                logger.info(`Error sending end-of-stream signal: ${e}`);
            }
        } catch (e) {
            logger.error(`Unexpected error in sender: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // HTTP synthesis
    // ------------------------------------------------------------------

    private async sendPayload(payload: Record<string, unknown>, format?: string): Promise<Buffer | null> {
        const url = `${this.apiUrl}${format ?? this.getFormat()}`;
        const res = await fetch(url, {
            method: "POST",
            headers: { "xi-api-key": this.apiKey, "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (res.ok) return Buffer.from(await res.arrayBuffer());
        logger.error(`Error: ${res.status} - ${await res.text()}`);
        return null;
    }

    private async generateHttp(text: string, format?: string): Promise<Buffer | null> {
        logger.info(`text ${text}`);
        return this.sendPayload(
            {
                text,
                model_id: this.model,
                voice_settings: {
                    stability: this.temperature,
                    similarity_boost: this.similarityBoost,
                    optimize_streaming_latency: 3,
                    speed: this.speed,
                    style: this.style,
                },
            },
            format
        );
    }

    override async synthesize(text: string): Promise<Buffer | null> {
        return this.generateHttp(text, "mp3_44100_128");
    }

    // ------------------------------------------------------------------
    // Generate
    // ------------------------------------------------------------------

    override async *generate(): AsyncGenerator<WsDataPacket> {
        let lastYieldTime: number | null = null;
        try {
            if (this.stream) {
                for await (const [message, textSynthesized] of this.receiver()) {
                    const genLoopStart = performance.now();
                    if (lastYieldTime) {
                        const loopGap = genLoopStart - lastYieldTime;
                        if (loopGap > 100) logger.info(`EL generate loop_gap=${loopGap.toFixed(0)}ms trace_id=${this.wsTraceId}`);
                    }
                    logger.info("Received message from server");

                    if (this.textQueue.length) {
                        this.metaInfo = this.textQueue.shift()!;
                        try {
                            if (this.currentTurnTtfb === null && this.wsSendTime !== null) {
                                this.currentTurnTtfb = (performance.now() - this.wsSendTime) / 1000;
                                this.metaInfo["synthesizer_latency"] = this.currentTurnTtfb;
                            }
                        } catch { /* ignore */ }
                    }

                    let audio: Buffer = message;

                    if (this.useMulaw) {
                        this.metaInfo!["format"] = "mulaw";
                    } else {
                        this.metaInfo!["format"] = "wav";
                        if (message.length !== 1 || message[0] !== 0x00) {
                            const procStart = performance.now();
                            const wav = await convertAudioToWav(message, "mp3");
                            audio = await resample(wav, parseInt(this.samplingRate, 10), { format: "wav" });
                            const procMs = performance.now() - procStart;
                            if (procMs > 50) logger.info(`EL audio_proc took=${procMs.toFixed(0)}ms trace_id=${this.wsTraceId}`);
                        }
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

                    this.metaInfo!["text_synthesized"] = textSynthesized;
                    this.metaInfo!["mark_id"] = uuidv4();
                    lastYieldTime = performance.now();
                    yield createWsDataPacket({ data: audio, metaInfo: this.metaInfo! });
                }
            } else {
                // HTTP non-streaming mode
                while (true) {
                    const message = (await this.internalQueue.get()) as Record<string, unknown>;
                    logger.info(`Generating TTS response for message: ${JSON.stringify(message)}, using mulaw ${this.useMulaw}`);

                    const metaInfo = message["meta_info"] as Record<string, unknown>;
                    const text = message["data"] as string;
                    let audio: Buffer | null = null;

                    if (this.caching && this.cache) {
                        const cached = this.cache.get(text) as Buffer | null;
                        if (cached) {
                            logger.info(`Cache hit for: ${text}`);
                            audio = cached;
                            metaInfo["is_cached"] = true;
                        } else {
                            const c = text.length;
                            this.synthesizedCharacters += c;
                            logger.info(`Not a cache hit, increasing characters by ${c}`);
                            metaInfo["is_cached"] = false;
                            audio = await this.generateHttp(text);
                            if (audio) this.cache.set(text, audio);
                        }
                    } else {
                        metaInfo["is_cached"] = false;
                        audio = await this.generateHttp(text);
                    }

                    metaInfo["text"] = text;

                    if (!this.firstChunkGenerated) {
                        metaInfo["is_first_chunk"] = true;
                        this.firstChunkGenerated = true;
                    }

                    if (metaInfo["end_of_llm_stream"]) {
                        metaInfo["end_of_synthesizer_stream"] = true;
                        this.firstChunkGenerated = false;
                    }

                    if (this.useMulaw) {
                        metaInfo["format"] = "mulaw";
                    } else {
                        metaInfo["format"] = "wav";
                        if (audio) {
                            const wav = await convertAudioToWav(audio, "mp3");
                            logger.info(`samplingRate ${this.samplingRate}`);
                            audio = await resample(wav, parseInt(this.samplingRate, 10), { format: "wav" });
                        }
                    }

                    yield createWsDataPacket({ data: audio ?? Buffer.alloc(0), metaInfo });
                }
            }
        } catch (e) {
            console.error(e);
            logger.info(`Error in eleven labs generate: ${e}`);
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
        logger.info("cleaning elevenlabs synthesizer tasks");

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

    async getSenderTask(): Promise<Promise<void> | null> {
        return this.senderTask;
    }
}