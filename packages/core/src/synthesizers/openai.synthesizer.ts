import OpenAI from "openai";
import { configureLogger } from "../helper/logger";
import { convertAudioToWav, createWsDataPacket, resample, type WsDataPacket } from "../helper/utils";
import { BaseSynthesizer } from "./base";

const logger = configureLogger("openaiSynthesizer");

export class OpenAISynthesizer extends BaseSynthesizer {
    private format: string;
    private voice: string;
    private sampleRate: number;
    private client: OpenAI;
    private model: string;
    private firstChunkGenerated = false;

    constructor(opts: {
        voice: string;
        audioFormat?: string;
        model?: string;
        stream?: boolean;
        samplingRate?: number | string;
        bufferSize?: number;
        synthesizerKey?: string;
        taskManagerInstance?: Record<string, unknown> | null;
        [key: string]: unknown;
    }) {
        super({
            taskManagerInstance: opts.taskManagerInstance ?? null,
            stream: opts.stream ?? false,
            bufferSize: opts.bufferSize ?? 400,
        });

        // Always mp3 — mulaw conversion for other formats breaks telephony
        this.format = "mp3";
        this.voice = opts.voice;
        this.model = opts.model ?? "tts-1";
        this.stream = false; // OpenAI synth always uses internal queue (non-streaming)

        this.sampleRate =
            typeof opts.samplingRate === "string"
                ? parseInt(opts.samplingRate, 10)
                : (opts.samplingRate ?? 8000);

        this.client = new OpenAI({
            apiKey: opts.synthesizerKey ?? process.env.OPENAI_API_KEY,
        });
    }

    // ------------------------------------------------------------------
    // HTTP generation
    // ------------------------------------------------------------------

    private async generateHttp(text: string): Promise<Buffer> {
        const response = await this.client.audio.speech.create({
            model: this.model,
            voice: this.voice as Parameters<typeof this.client.audio.speech.create>[0]["voice"],
            response_format: this.format as "mp3",
            input: text,
        });

        const chunks: Buffer[] = [];
        for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
            chunks.push(Buffer.from(chunk));
        }
        return Buffer.concat(chunks);
    }

    private async *generateStream(text: string): AsyncGenerator<Buffer> {
        const response = await this.client.audio.speech.create({
            model: this.model,
            voice: this.voice as Parameters<typeof this.client.audio.speech.create>[0]["voice"],
            response_format: "mp3",
            input: text,
        });

        for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
            yield Buffer.from(chunk);
        }
    }

    // ------------------------------------------------------------------
    // Public synthesize (one-off, e.g. voice lab / IVR)
    // ------------------------------------------------------------------

    override async synthesize(text: string): Promise<Buffer> {
        return this.generateHttp(text);
    }

    // ------------------------------------------------------------------
    // Generate loop (consumes internal queue)
    // ------------------------------------------------------------------

    override async *generate(): AsyncGenerator<WsDataPacket> {
        try {
            while (true) {
                const message = (await this.internalQueue.get()) as Record<string, unknown>;
                logger.info(`Generating TTS response for message: ${JSON.stringify(message)}`);

                const metaInfo = message["meta_info"] as Record<string, unknown>;
                const text = message["data"] as string;
                metaInfo["text"] = text;

                if (!this.shouldSynthesizeResponse(metaInfo["sequence_id"] as number)) {
                    logger.info(
                        `Not synthesizing: sequence_id ${metaInfo["sequence_id"]} not in current ids`
                    );
                    return;
                }

                if (this.stream) {
                    for await (const chunk of this.generateStream(text)) {
                        if (!this.firstChunkGenerated) {
                            metaInfo["is_first_chunk"] = true;
                            this.firstChunkGenerated = true;
                        }
                        const wav = await convertAudioToWav(chunk, "mp3");
                        const resampled = await resample(wav, this.sampleRate, { format: "wav" });
                        yield createWsDataPacket({ data: resampled, metaInfo });
                    }

                    if (metaInfo["end_of_llm_stream"]) {
                        metaInfo["end_of_synthesizer_stream"] = true;
                        this.firstChunkGenerated = false;
                        yield createWsDataPacket({ data: Buffer.from([0x00]), metaInfo });
                    }
                } else {
                    logger.info("Generating without a stream");
                    const audio = await this.generateHttp(text);

                    if (!this.firstChunkGenerated) {
                        metaInfo["is_first_chunk"] = true;
                        this.firstChunkGenerated = true;
                    }

                    if (metaInfo["end_of_llm_stream"]) {
                        metaInfo["end_of_synthesizer_stream"] = true;
                        this.firstChunkGenerated = false;
                    }

                    const wav = await convertAudioToWav(audio, "mp3");
                    const resampled = await resample(wav, this.sampleRate, { format: "wav" });
                    yield createWsDataPacket({ data: resampled, metaInfo });
                }
            }
        } catch (e) {
            logger.error(`Error in openai generate: ${e}`);
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