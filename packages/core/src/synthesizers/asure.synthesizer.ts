import * as sdk from "microsoft-cognitiveservices-speech-sdk";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket, type WsDataPacket } from "../helper/utils";
import { InmemoryScalarCache } from "../memory/cache/inmemory.cache";
import { BaseSynthesizer } from "./base";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("azureSynthesizer");

// Escape XML special characters (replaces xml.sax.saxutils.escape)
function xmlEscape(text: string): string {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");
}

export class AzureSynthesizer extends BaseSynthesizer {
    private model: string;
    private language: string;
    private voice: string;
    private sampleRate: string;
    private firstChunkGenerated = false;
    private synthesizedCharacters = 0;
    private caching: boolean;
    private speed: number | null;
    private cache: InmemoryScalarCache | null = null;
    private subscriptionKey: string;
    private region: string;
    private speechConfig: sdk.SpeechConfig;
    private latencyStats = {
        request_count: 0,
        total_first_byte_latency: 0,
        min_latency: Infinity,
        max_latency: 0,
    };

    constructor(opts: {
        voice: string;
        language: string;
        model?: string;
        stream?: boolean;
        samplingRate?: number | string;
        bufferSize?: number;
        caching?: boolean;
        speed?: number | null;
        synthesizerKey?: string;
        region?: string;
        taskManagerInstance?: Record<string, unknown> | null;
        [key: string]: unknown;
    }) {
        super({
            taskManagerInstance: opts.taskManagerInstance ?? null,
            stream: opts.stream ?? false,
            bufferSize: opts.bufferSize ?? 150,
        });

        this.model = opts.model ?? "neural";
        this.language = opts.language;
        this.voice = `${opts.language}-${opts.voice}${this.model}`;
        this.sampleRate = String(opts.samplingRate ?? 8000);
        this.stream = opts.stream ?? false;
        this.speed = opts.speed ?? null;
        this.caching = opts.caching ?? true;

        if (this.caching) this.cache = new InmemoryScalarCache();

        this.subscriptionKey = opts.synthesizerKey ?? process.env.AZURE_SPEECH_KEY!;
        this.region = (opts.region as string | undefined) ?? process.env.AZURE_SPEECH_REGION!;

        this.speechConfig = sdk.SpeechConfig.fromSubscription(this.subscriptionKey, this.region);
        this.speechConfig.speechSynthesisOutputFormat =
            sdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm;
        this.speechConfig.speechSynthesisVoiceName = this.voice;

        logger.info(`${this.voice} initialized`);
    }

    // ------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------

    override getSynthesizedCharacters(): number {
        return this.synthesizedCharacters;
    }

    override getEngine(): string {
        return this.model;
    }

    override supportsWebsocket(): boolean {
        return false;
    }

    override getSleepTime(): number {
        return 0.01;
    }

    // ------------------------------------------------------------------
    // SSML builder
    // ------------------------------------------------------------------

    private buildSsml(text: string): string | null {
        if (!this.speed || this.speed === 1) return null;
        const body = xmlEscape(text);
        const prosodyAttrs = this.speed != null ? `rate="${this.speed}"` : "";
        return `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="${this.language}">
  <voice name="${this.voice}">
    <prosody ${prosodyAttrs}>${body}</prosody>
  </voice>
</speak>`;
    }

    // ------------------------------------------------------------------
    // HTTP (one-off) synthesis
    // ------------------------------------------------------------------

    override async synthesize(text: string): Promise<Buffer | null> {
        return this.generateHttp(text);
    }

    private generateHttp(text: string): Promise<Buffer | null> {
        return new Promise((resolve, reject) => {
            const synthesizer = new sdk.SpeechSynthesizer(this.speechConfig, undefined);
            const ssml = this.buildSsml(text);

            const handleResult = (result: sdk.SpeechSynthesisResult) => {
                synthesizer.close();
                if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
                    resolve(Buffer.from(result.audioData));
                } else if (result.reason === sdk.ResultReason.Canceled) {
                    const details = sdk.CancellationDetails.fromResult(result);
                    this.logCancellationError(details);
                    reject(new Error(`Azure TTS canceled: ${details.errorDetails}`));
                } else {
                    reject(new Error(`Azure TTS failed: ${result.reason}`));
                }
            };

            if (ssml) {
                synthesizer.speakSsmlAsync(ssml, handleResult, (e) => { synthesizer.close(); reject(e); });
            } else {
                synthesizer.speakTextAsync(text, handleResult, (e) => { synthesizer.close(); reject(e); });
            }
        });
    }

    private logCancellationError(details: sdk.CancellationDetails): void {
        logger.error(`Azure TTS canceled: ${details.reason}`);
        if (details.reason === sdk.CancellationReason.Error) {
            logger.error(`Azure TTS error details: ${details.errorDetails}`);
            const code = details.ErrorCode;
            if (code === sdk.CancellationErrorCode.AuthenticationFailure) {
                logger.error(`Azure TTS authentication failed - Region: ${this.region}`);
            } else if (code === sdk.CancellationErrorCode.Forbidden) {
                logger.error(`Azure TTS forbidden - Region: ${this.region}`);
            } else if (code === sdk.CancellationErrorCode.BadRequestParameters) {
                logger.error(`Azure TTS bad request - Region: ${this.region}`);
            } else if (code === sdk.CancellationErrorCode.ConnectionFailure) {
                logger.error(`Azure TTS connection failure - Region: ${this.region}`);
            }
        }
    }

    // ------------------------------------------------------------------
    // Streaming generate loop
    // ------------------------------------------------------------------

    override async *generate(): AsyncGenerator<WsDataPacket> {
        try {
            while (true) {
                const message = (await this.internalQueue.get()) as Record<string, unknown>;
                logger.info(`Generating TTS response for message: ${JSON.stringify(message)}`);

                const metaInfo = message["meta_info"] as Record<string, unknown>;
                const text = message["data"] as string;

                if (!this.shouldSynthesizeResponse(metaInfo["sequence_id"] as number)) {
                    logger.info(`Not synthesizing: sequence_id ${metaInfo["sequence_id"]} not in current ids`);
                    return;
                }

                // Cache hit
                if (this.caching && this.cache) {
                    const cached = this.cache.get(text) as Buffer | null;
                    if (cached) {
                        logger.info(`Cache hit for: ${text}`);
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
                        metaInfo["format"] = "wav";
                        metaInfo["text_synthesized"] = `${text} `;
                        metaInfo["mark_id"] = uuidv4();
                        yield createWsDataPacket({ data: cached, metaInfo });
                        continue;
                    }
                }

                // Create synthesizer per-request to avoid blocking
                let synthesizer: sdk.SpeechSynthesizer;
                try {
                    synthesizer = new sdk.SpeechSynthesizer(this.speechConfig, undefined);
                } catch (e) {
                    logger.error(`Failed to create Azure TTS synthesizer: ${e}`);
                    continue;
                }

                const startTime = performance.now();
                metaInfo["synthesizer_start_time"] = startTime;
                const fullAudio: Buffer[] = [];

                // Bridge the SDK's callback-based streaming to an async generator
                // using a Promise queue (same pattern as SarvamSynthesizer)
                const chunkResolvers: ((chunk: Buffer | null) => void)[] = [];
                const pendingChunks: (Buffer | null)[] = [];
                let synthesisError: string | null = null;

                const pushChunk = (chunk: Buffer | null) => {
                    const resolver = chunkResolvers.shift();
                    if (resolver) resolver(chunk);
                    else pendingChunks.push(chunk);
                };

                const nextChunk = (): Promise<Buffer | null> =>
                    new Promise((resolve) => {
                        const pending = pendingChunks.shift();
                        if (pending !== undefined) resolve(pending);
                        else chunkResolvers.push(resolve);
                    });

                // SDK event: streaming chunk
                synthesizer.synthesizing = (_s, evt) => {
                    try {
                        const data = Buffer.from(evt.result.audioData);
                        if (data.length) pushChunk(data);
                    } catch (e) {
                        logger.error(`Error in synthesizing handler: ${e}`);
                    }
                };

                // SDK event: synthesis complete
                synthesizer.synthesisCompleted = (_s, evt) => {
                    if (evt.result.reason === sdk.ResultReason.Canceled) {
                        const details = sdk.CancellationDetails.fromResult(evt.result);
                        this.logCancellationError(details);
                        synthesisError = details.errorDetails;
                    }
                    pushChunk(null); // Signal end
                };

                // SDK event: cancelled
                synthesizer.SynthesisCanceled = (_s, evt) => {
                    const details = sdk.CancellationDetails.fromResult(evt.result);
                    synthesisError = details.errorDetails;
                    logger.error(`Azure TTS canceled: ${synthesisError}`);
                    pushChunk(null);
                };

                // Start async synthesis (non-blocking)
                const ssml = this.buildSsml(text);
                try {
                    if (ssml) {
                        synthesizer.speakSsmlAsync(ssml, () => { }, (e) => {
                            logger.error(`Azure TTS synthesis error: ${e}`);
                            pushChunk(null);
                        });
                    } else {
                        synthesizer.speakTextAsync(text, () => { }, (e) => {
                            logger.error(`Azure TTS synthesis error: ${e}`);
                            pushChunk(null);
                        });
                    }
                } catch (e) {
                    logger.error(`Failed to start Azure TTS synthesis: ${e}`);
                    synthesizer.close();
                    continue;
                }

                logger.info(`Azure TTS request sent for ${text.length} chars`);

                // Drain chunks
                while (true) {
                    const chunk = await nextChunk();
                    if (chunk === null) break;
                    if (synthesisError) break;

                    if (this.caching) fullAudio.push(chunk);

                    // First chunk latency
                    if (!this.firstChunkGenerated) {
                        const firstChunkMs = Math.round(performance.now() - startTime);
                        this.latencyStats.request_count += 1;
                        this.latencyStats.total_first_byte_latency += firstChunkMs;
                        this.latencyStats.min_latency = Math.min(this.latencyStats.min_latency, firstChunkMs);
                        this.latencyStats.max_latency = Math.max(this.latencyStats.max_latency, firstChunkMs);
                        if (!("synthesizer_first_result_latency" in metaInfo)) {
                            const lat = (performance.now() - (metaInfo["synthesizer_start_time"] as number ?? startTime)) / 1000;
                            metaInfo["synthesizer_first_result_latency"] = lat;
                            metaInfo["synthesizer_latency"] = lat;
                        }
                        metaInfo["is_first_chunk"] = true;
                        this.firstChunkGenerated = true;
                    } else {
                        metaInfo["is_first_chunk"] = false;
                    }

                    metaInfo["text"] = text;
                    metaInfo["format"] = "wav";
                    metaInfo["text_synthesized"] = `${text} `;
                    metaInfo["mark_id"] = uuidv4();
                    yield createWsDataPacket({ data: chunk, metaInfo });
                }

                // Final chunk
                if (metaInfo["end_of_llm_stream"]) {
                    metaInfo["end_of_synthesizer_stream"] = true;
                    this.firstChunkGenerated = false;
                }
                metaInfo["synthesizer_total_stream_duration"] =
                    (performance.now() - (metaInfo["synthesizer_start_time"] as number ?? startTime)) / 1000;

                // Cache complete audio
                if (this.caching && this.cache && fullAudio.length) {
                    logger.info(`Caching audio for: ${text}`);
                    this.cache.set(text, Buffer.concat(fullAudio));
                }

                this.synthesizedCharacters += text.length;
                synthesizer.close();
            }
        } catch (e) {
            if ((e as Error).message?.includes("cancel")) {
                logger.info("Azure synthesizer task cancelled - shutting down cleanly");
            } else {
                logger.error(`Error in Azure TTS generate: ${e}`);
                throw e;
            }
        }
    }

    // ------------------------------------------------------------------
    // Push / connection
    // ------------------------------------------------------------------
    override push(text: string): void {
        const message = {
            data: text
        };

        logger.info(`Pushed message to internal queue ${JSON.stringify(message)}`);
        this.internalQueue.put_nowait(message);
    }

    async openConnection(): Promise<void> { }
}