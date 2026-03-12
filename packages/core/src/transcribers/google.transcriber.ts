import { Worker, isMainThread, parentPort, workerData } from "worker_threads";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket } from "../helper/utils";
import { BaseTranscriber } from "./base";
import * as speech from "@google-cloud/speech";

const logger = configureLogger("googleTranscriber");

// ------------------------------------------------------------------
// Worker thread: runs the blocking gRPC stream
// ------------------------------------------------------------------
if (!isMainThread) {
    const { encoding, sampleRateHertz, language, model } = workerData as {
        encoding: string;
        sampleRateHertz: number;
        language: string;
        model: string;
    };

    const client = new speech.SpeechClient();

    // Encoding map
    const encUpper = (encoding ?? "").toUpperCase();
    let encodingEnum: speech.protos.google.cloud.speech.v1p1beta1.RecognitionConfig.AudioEncoding =
        speech.protos.google.cloud.speech.v1p1beta1.RecognitionConfig.AudioEncoding.LINEAR16;
    if (encUpper.includes("MULAW") || encUpper.includes("ULAW")) {
        encodingEnum =
            speech.protos.google.cloud.speech.v1p1beta1.RecognitionConfig.AudioEncoding.MULAW;
    }

    const recognizeStream = client
        .streamingRecognize({
            config: {
                encoding: encodingEnum,
                sampleRateHertz,
                languageCode: language,
                model,
                enableAutomaticPunctuation: true,
                maxAlternatives: 1,
            },
            interimResults: true,
            singleUtterance: false,
        })
        .on("data", (response: speech.protos.google.cloud.speech.v1p1beta1.IStreamingRecognizeResponse) => {
            if (!response.results?.length) return;
            const result = response.results[0]!;
            const isFinal = result.isFinal ?? false;
            const transcript = (result.alternatives?.[0]?.transcript ?? "").trim();
            if (transcript) {
                parentPort!.postMessage({ type: isFinal ? "final" : "interim", transcript });
            }
        })
        .on("error", (err: Error) => {
            parentPort!.postMessage({ type: "error", message: err.message });
        })
        .on("end", () => {
            parentPort!.postMessage({ type: "end" });
        });

    // Receive audio chunks from main thread
    parentPort!.on("message", (msg: { type: string; data?: Buffer }) => {
        if (msg.type === "audio" && msg.data) {
            recognizeStream.write(msg.data);
        } else if (msg.type === "end") {
            recognizeStream.end();
        }
    });
}

// ------------------------------------------------------------------
// Main thread: GoogleTranscriber class
// ------------------------------------------------------------------
export class GoogleTranscriber extends BaseTranscriber {
    private provider: string;
    private transcriberOutputQueue: { put: (data: unknown) => Promise<void> };
    private language: string;
    private model: string;
    private runId: string;
    private encoding: string;
    private sampleRateHertz: number;
    private audioFrameDuration = 0.0;
    private numFrames = 0;
    private audioSubmitted = false;
    private audioSubmissionTime: number | null = null;
    private requestId: string | null = null;
    private connectionStartTime: number | null = null;
    private connectionAuthenticated = false;
    private transcriptionTask: Promise<void> | null = null;
    private currentTurnStartTime: number | null = null;
    private currentTurnId: string | number | null = null;
    private running = false;
    private worker: Worker | null = null;

    constructor(opts: {
        telephonyProvider: string;
        inputQueue?: { get: () => Promise<unknown> } | null;
        outputQueue: { put: (data: unknown) => Promise<void> };
        language?: string;
        encoding?: string;
        sampleRateHertz?: number;
        model?: string;
        runId?: string;
        [key: string]: unknown;
    }) {
        super(opts.inputQueue ?? null);

        this.provider = opts.telephonyProvider ?? "";
        this.transcriberOutputQueue = opts.outputQueue;
        this.language = opts.language ?? "en-US";
        this.model = opts.model ?? "latest_long";
        this.runId = (opts.runId as string | undefined) ?? "";
        this.metaInfo = {};

        // Provider-specific audio config
        if (["twilio", "exotel", "plivo", "vobiz"].includes(this.provider)) {
            this.encoding = this.provider === "twilio" ? "MULAW" : "LINEAR16";
            this.sampleRateHertz = 8000;
            this.audioFrameDuration = 0.2;
        } else if (this.provider === "web_based_call") {
            this.encoding = "LINEAR16";
            this.sampleRateHertz = 16000;
            this.audioFrameDuration = 0.256;
        } else if (this.provider === "playground") {
            this.encoding = "LINEAR16";
            this.sampleRateHertz = 8000;
            this.audioFrameDuration = 0.0;
        } else {
            this.encoding = (opts.encoding as string | undefined) ?? "LINEAR16";
            this.sampleRateHertz = (opts.sampleRateHertz as number | undefined) ?? 16000;
        }
    }

    // ------------------------------------------------------------------
    // Enqueue helper — called from both async and sync contexts
    // ------------------------------------------------------------------
    private async enqueueOutput(data: unknown, meta?: Record<string, unknown>): Promise<void> {
        const packet = createWsDataPacket({ data, metaInfo: meta ?? this.metaInfo ?? {} });
        try {
            await this.transcriberOutputQueue.put(packet);
        } catch (e) {
            logger.error(`Failed to enqueue packet: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Validate Google client
    // ------------------------------------------------------------------
    async googleConnect(): Promise<void> {
        try {
            const start = performance.now();
            // Instantiating SpeechClient validates ADC credentials
            new speech.SpeechClient();
            this.connectionAuthenticated = true;
            if (!this.connectionTime) {
                this.connectionTime = Math.round(performance.now() - start);
            }
            logger.info("Successfully validated Google Speech client");
        } catch (e) {
            logger.error(`Failed to validate Google Speech client: ${e}`);
            throw new Error(`Failed to validate Google Speech client: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // run()
    // ------------------------------------------------------------------
    async run(): Promise<void> {
        try {
            await this.googleConnect();
            this.running = true;
            this.transcriptionTask = this.transcribeWrapper();
            await this.transcriptionTask;
        } catch (e) {
            logger.error(`Error starting GoogleTranscriber: ${e}`);
            await this.toggleConnection();
        }
    }

    // ------------------------------------------------------------------
    // Transcribe wrapper — spawns worker + sends audio
    // ------------------------------------------------------------------
    private async transcribeWrapper(): Promise<void> {
        try {
            // Spawn worker thread for blocking gRPC stream
            this.worker = new Worker(__filename, {
                workerData: {
                    encoding: this.encoding,
                    sampleRateHertz: this.sampleRateHertz,
                    language: this.language,
                    model: this.model,
                },
            });

            // Wire worker messages → output queue
            this.worker.on("message", async (msg: { type: string; transcript?: string; message?: string }) => {
                const now = performance.now();

                if (msg.type === "interim" && msg.transcript) {
                    try {
                        if (this.metaInfo && !("transcriber_first_result_latency" in this.metaInfo) && "transcriber_start_time" in this.metaInfo) {
                            this.metaInfo["transcriber_first_result_latency"] =
                                (now - (this.metaInfo["transcriber_start_time"] as number)) / 1000;
                        }
                    } catch { /* ignore */ }
                    await this.enqueueOutput(
                        { type: "interim_transcript_received", content: msg.transcript },
                        this.metaInfo ?? {}
                    );

                } else if (msg.type === "final" && msg.transcript) {
                    try {
                        if (this.metaInfo && "transcriber_start_time" in this.metaInfo) {
                            if (!("transcriber_first_result_latency" in this.metaInfo)) {
                                this.metaInfo["transcriber_first_result_latency"] =
                                    (now - (this.metaInfo["transcriber_start_time"] as number)) / 1000;
                            }
                            this.metaInfo["transcriber_total_stream_duration"] =
                                (now - (this.metaInfo["transcriber_start_time"] as number)) / 1000;
                        }
                    } catch { /* ignore */ }

                    this.appendTurnLatency();
                    await this.enqueueOutput(
                        { type: "transcript", content: msg.transcript },
                        this.metaInfo ?? {}
                    );

                } else if (msg.type === "error") {
                    logger.error(`Google streaming error: ${msg.message}`);
                    const errMeta = { ...(this.metaInfo ?? {}), error: msg.message, error_type: "streaming_error" };
                    await this.enqueueOutput("transcriber_connection_closed", errMeta);

                } else if (msg.type === "end") {
                    const closedMeta = { ...(this.metaInfo ?? {}) };
                    if (!("transcriber_total_stream_duration" in closedMeta) && "transcriber_start_time" in closedMeta) {
                        closedMeta["transcriber_total_stream_duration"] =
                            (performance.now() - (closedMeta["transcriber_start_time"] as number)) / 1000;
                    }
                    await this.enqueueOutput("transcriber_connection_closed", closedMeta);
                }
            });

            this.worker.on("error", async (err) => {
                logger.error(`Worker error: ${err}`);
                await this.enqueueOutput("transcriber_connection_closed", {
                    ...(this.metaInfo ?? {}), error: err.message, error_type: "worker_error",
                });
            });

            // Send audio from input_queue → worker
            await this.sendAudioToTranscriber();

        } catch (e) {
            logger.error(`Error in transcribeWrapper: ${e}`);
            await this.toggleConnection();
        }
    }

    // ------------------------------------------------------------------
    // Read input_queue → forward to worker
    // ------------------------------------------------------------------
    private async sendAudioToTranscriber(): Promise<void> {
        try {
            while (true) {
                const wsDataPacket = (await (this.inputQueue as { get: () => Promise<Record<string, unknown>> }).get());

                // Initialize metadata on first packet
                if (!this.audioSubmitted) {
                    this.audioSubmitted = true;
                    this.audioSubmissionTime = Date.now() / 1000;
                    this.requestId = BaseTranscriber.generateRequestId();
                    this.metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                    this.metaInfo["request_id"] = this.requestId;
                    try {
                        this.metaInfo["transcriber_start_time"] = performance.now();
                        this.currentTurnStartTime = this.metaInfo["transcriber_start_time"] as number;
                        this.currentTurnId =
                            (this.metaInfo["turn_id"] ?? this.metaInfo["request_id"] ?? this.requestId) as string;
                    } catch { /* ignore */ }
                }

                // EOS sentinel
                const metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                if (metaInfo["eos"] === true) {
                    this.worker?.postMessage({ type: "end" });
                    break;
                }

                // Forward audio bytes to worker
                let data = wsDataPacket["data"];
                if (data) {
                    this.numFrames++;
                    let buf: Buffer;

                    if (typeof data === "string") {
                        try {
                            buf = Buffer.from(data, "base64");
                        } catch {
                            buf = Buffer.from(data, "utf-8");
                        }
                    } else {
                        buf = data as Buffer;
                    }

                    this.worker?.postMessage({ type: "audio", data: buf }, [new ArrayBuffer(buf.buffer.byteLength)]);
                }
            }
        } catch (e) {
            logger.error(`Error in sendAudioToTranscriber: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Turn latency tracking
    // ------------------------------------------------------------------
    private appendTurnLatency(): void {
        try {
            if (this.currentTurnId && this.currentTurnStartTime && this.metaInfo) {
                const firstMs = Math.round(((this.metaInfo["transcriber_first_result_latency"] as number) ?? 0) * 1000);
                const totalMs = Math.round((performance.now() - this.currentTurnStartTime));
                this.turnLatencies.push({
                    turn_id: this.currentTurnId,
                    sequence_id: this.currentTurnId,
                    first_result_latency_ms: firstMs,
                    total_stream_duration_ms: totalMs,
                });
                this.metaInfo["turn_latencies"] = this.turnLatencies;
                this.currentTurnStartTime = null;
                this.currentTurnId = null;
            }
        } catch (e) {
            logger.error(`Error appending turn latency: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // toggle_connection / cleanup
    // ------------------------------------------------------------------
    async toggleConnection(): Promise<void> {
        logger.info("toggleConnection called on GoogleTranscriber");
        this.running = false;
        this.connectionAuthenticated = false;

        this.worker?.postMessage({ type: "end" });

        await new Promise<void>((resolve) => {
            if (!this.worker) { resolve(); return; }
            const timer = setTimeout(() => {
                logger.warn("Worker did not exit within timeout");
                resolve();
            }, 2_000);
            this.worker!.once("exit", () => { clearTimeout(timer); resolve(); });
        });

        this.worker = null;
        logger.info("GoogleTranscriber connection toggled off");
    }

    override async cleanup(): Promise<void> {
        logger.info("Cleaning up Google transcriber resources");
        this.running = false;
        this.connectionAuthenticated = false;

        this.worker?.postMessage({ type: "end" });

        await new Promise<void>((resolve) => {
            if (!this.worker) { resolve(); return; }
            const timer = setTimeout(resolve, 2_000);
            this.worker!.once("exit", () => { clearTimeout(timer); resolve(); });
        });

        this.worker = null;
        this.connectionStartTime = null;
        this.connectionTime = null;
        logger.info("Google transcriber cleanup complete");
    }

    getMetaInfo(): Record<string, unknown> {
        return this.metaInfo ?? {};
    }
}