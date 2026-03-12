import * as sdk from "microsoft-cognitiveservices-speech-sdk";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket, timestampMs } from "../helper/utils";
import { BaseTranscriber } from "./base";

const logger = configureLogger("azureTranscriber");

export class AzureTranscriber extends BaseTranscriber {
    private transcriptionTask: Promise<void> | null = null;
    private subscriptionKey: string;
    private serviceRegion: string;
    private pushStream: sdk.PushAudioInputStream | null = null;
    private recognizer: sdk.SpeechRecognizer | null = null;
    private transcriberOutputQueue: { put: (data: unknown) => Promise<void> };
    private audioSubmitted = false;
    private audioSubmissionTime: number | null = null;
    private sendAudioTask: Promise<void> | null = null;
    private recognitionLanguage: string;
    private audioProvider: string;
    private channels = 1;
    private encoding: string;
    private samplingRate: number;
    private bitsPerSample: number;
    private runId: string;
    private duration = 0;
    private startTime: number | null = null;
    private endTime: number | null = null;

    private audioFrameTimestamps: [number, number, number][] = [];
    private numFrames = 0;
    private audioFrameDuration = 0.0;

    private currentTurnInterimDetails: Record<string, unknown>[] = [];
    private currentTurnStartTime: number | null = null;
    private currentTurnId: string | null = null;

    constructor(opts: {
        telephonyProvider: string;
        inputQueue?: { get: () => Promise<unknown> } | null;
        outputQueue: { put: (data: unknown) => Promise<void> };
        language?: string;
        encoding?: string;
        runId?: string;
        [key: string]: unknown;
    }) {
        super(opts.inputQueue ?? null);

        this.subscriptionKey = process.env.AZURE_SPEECH_KEY!;
        this.serviceRegion = process.env.AZURE_SPEECH_REGION!;
        this.transcriberOutputQueue = opts.outputQueue;
        this.recognitionLanguage = opts.language ?? "en-US";
        this.audioProvider = opts.telephonyProvider;
        this.encoding = "linear16";
        this.samplingRate = 8000;
        this.bitsPerSample = 16;
        this.runId = (opts.runId as string | undefined) ?? "";
        this.metaInfo = {};

        if (["twilio", "exotel", "plivo"].includes(this.audioProvider)) {
            this.encoding = this.audioProvider === "twilio" ? "mulaw" : "linear16";
            if (this.encoding === "mulaw") this.bitsPerSample = 8;
            this.audioFrameDuration = 0.2;
        } else if (this.audioProvider === "web_based_call") {
            this.samplingRate = 16000;
            this.audioFrameDuration = 0.256;
        }

        if (this.audioFrameDuration === 0.0) this.audioFrameDuration = 0.2;
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
    // Ticks helper (Azure offset/duration are in 100-nanosecond ticks)
    // ------------------------------------------------------------------

    private ticksToSeconds(ticks: number): number {
        return ticks / 10_000_000;
    }

    // ------------------------------------------------------------------
    // EOS check
    // ------------------------------------------------------------------

    private checkAndProcessEndOfStream(wsDataPacket: Record<string, unknown>): boolean {
        const metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
        if (metaInfo["eos"] === true) {
            logger.info("End of stream detected");
            this.syncCleanup();
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Initialize Azure connection
    // ------------------------------------------------------------------

    async initializeConnection(): Promise<void> {
        try {
            const speechConfig = sdk.SpeechConfig.fromSubscription(this.subscriptionKey, this.serviceRegion);
            speechConfig.speechRecognitionLanguage = this.recognitionLanguage;

            const waveFormat = this.encoding === "mulaw"
                ? sdk.AudioFormatTag.MuLaw
                : sdk.AudioFormatTag.PCM;

            const audioFormat = sdk.AudioStreamFormat.getWaveFormatPCM(
                this.samplingRate,
                this.bitsPerSample,
                this.channels
            );
            // Note: Azure SDK JS uses getWaveFormatPCM; mulaw requires custom format via getCompressedFormat
            // For mulaw, override:
            const streamFormat = this.encoding === "mulaw"
                ? sdk.AudioStreamFormat.getWaveFormat(
                    this.samplingRate,
                    this.bitsPerSample,
                    this.channels,
                    sdk.AudioFormatTag.MuLaw
                )
                : audioFormat;

            this.pushStream = sdk.AudioInputStream.createPushStream(streamFormat);
            const audioConfig = sdk.AudioConfig.fromStreamInput(this.pushStream);
            this.recognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

            // Wire SDK events — callbacks fire on SDK threads, we post to async queue
            this.recognizer.recognizing = (_s, evt) => {
                this.recognizingHandler(evt).catch((e) =>
                    logger.error(`recognizingHandler error: ${e}`)
                );
            };

            this.recognizer.recognized = (_s, evt) => {
                this.recognizedHandler(evt).catch((e) =>
                    logger.error(`recognizedHandler error: ${e}`)
                );
            };

            this.recognizer.canceled = (_s, evt) => {
                this.canceledHandler(evt).catch((e) =>
                    logger.error(`canceledHandler error: ${e}`)
                );
            };

            this.recognizer.sessionStarted = (_s, evt) => {
                this.sessionStartedHandler(evt).catch((e) =>
                    logger.error(`sessionStartedHandler error: ${e}`)
                );
            };

            this.recognizer.sessionStopped = (_s, evt) => {
                this.sessionStoppedHandler(evt).catch((e) =>
                    logger.error(`sessionStoppedHandler error: ${e}`)
                );
            };

            const start = performance.now();
            await new Promise<void>((resolve, reject) => {
                this.recognizer!.startContinuousRecognitionAsync(
                    () => resolve(),
                    (err) => reject(new Error(err))
                );
            });

            logger.info("Azure speech recognition started successfully");
            if (!this.connectionTime) {
                this.connectionTime = Math.round(performance.now() - start);
            }
        } catch (e) {
            logger.error(`Error in initializeConnection: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Send audio loop
    // ------------------------------------------------------------------

    private async sendAudioToTranscriber(): Promise<void> {
        try {
            while (true) {
                const wsDataPacket = (await (this.inputQueue as { get: () => Promise<Record<string, unknown>> }).get());

                if (!this.audioSubmitted) {
                    this.metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                    this.audioSubmitted = true;
                    this.audioSubmissionTime = Date.now() / 1000;
                    this.currentRequestId = BaseTranscriber.generateRequestId();
                    this.metaInfo["request_id"] = this.currentRequestId;
                    try {
                        this.metaInfo["transcriber_start_time"] = performance.now();
                    } catch { /* ignore */ }
                }

                if (this.checkAndProcessEndOfStream(wsDataPacket)) break;

                const data = wsDataPacket["data"] as Buffer | undefined;
                if (data) {
                    const frameStart = this.numFrames * this.audioFrameDuration;
                    const frameEnd = (this.numFrames + 1) * this.audioFrameDuration;
                    const sendTimestamp = timestampMs();
                    this.audioFrameTimestamps.push([frameStart, frameEnd, sendTimestamp]);
                    this.numFrames++;
                    this.pushStream?.write(Buffer.from(data).buffer);
                }
            }
        } catch (e) {
            logger.error(`Error in sendAudioToTranscriber: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // SDK event handlers
    // ------------------------------------------------------------------

    private async recognizingHandler(evt: sdk.SpeechRecognitionEventArgs): Promise<void> {
        const text = evt.result.text?.trim();
        logger.info(`Intermediate results: ${text} | run_id - ${this.runId}`);
        if (!text) return;

        const offsetSec = this.ticksToSeconds(evt.result.offset);
        const durationSec = this.ticksToSeconds(evt.result.duration);
        const audioPositionEnd = offsetSec + durationSec;

        let latencyMs: number | null = null;
        const audioSentAt = this.findAudioSendTimestamp(audioPositionEnd);
        if (audioSentAt !== null) {
            latencyMs = Math.round((timestampMs() - audioSentAt) * 100000) / 100000;
        }

        this.currentTurnInterimDetails.push({
            transcript: text,
            latency_ms: latencyMs,
            is_final: false,
            received_at: Date.now() / 1000,
            offset_seconds: offsetSec,
            duration_seconds: durationSec,
        });

        try {
            if (this.metaInfo &&
                "transcriber_start_time" in this.metaInfo &&
                !("transcriber_first_result_latency" in this.metaInfo)) {
                this.metaInfo["transcriber_first_result_latency"] =
                    (performance.now() - (this.metaInfo["transcriber_start_time"] as number)) / 1000;
                if (latencyMs !== null) {
                    this.metaInfo["transcriber_latency"] = latencyMs / 1000;
                }
            }
        } catch { /* ignore */ }

        await this.transcriberOutputQueue.put(
            createWsDataPacket({
                data: { type: "interim_transcript_received", content: text },
                metaInfo: this.metaInfo ?? {},
            })
        );
    }

    private async recognizedHandler(evt: sdk.SpeechRecognitionEventArgs): Promise<void> {
        const text = evt.result.text?.trim();
        logger.info(`Final transcript: ${text} | run_id - ${this.runId}`);
        if (!text) return;

        const offsetSec = this.ticksToSeconds(evt.result.offset);
        const durationSec = this.ticksToSeconds(evt.result.duration);
        const audioPositionEnd = offsetSec + durationSec;

        let latencyMs: number | null = null;
        const audioSentAt = this.findAudioSendTimestamp(audioPositionEnd);
        if (audioSentAt !== null) {
            latencyMs = Math.round((timestampMs() - audioSentAt) * 100000) / 100000;
        }

        this.currentTurnInterimDetails.push({
            transcript: text,
            latency_ms: latencyMs,
            is_final: true,
            received_at: Date.now() / 1000,
            offset_seconds: offsetSec,
            duration_seconds: durationSec,
        });

        try {
            if (!this.currentTurnId) {
                this.currentTurnId = (this.metaInfo?.["turn_id"] ?? this.metaInfo?.["request_id"]) as string | null;
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
            });

            this.currentTurnInterimDetails = [];
            this.currentTurnStartTime = null;
            this.currentTurnId = null;
        } catch (e) {
            logger.error(`Error tracking turn latencies: ${e}`);
        }

        try {
            if (this.metaInfo && "transcriber_start_time" in this.metaInfo) {
                this.metaInfo["transcriber_total_stream_duration"] =
                    (performance.now() - (this.metaInfo["transcriber_start_time"] as number)) / 1000;
                if (latencyMs !== null) {
                    this.metaInfo["transcriber_latency"] = latencyMs / 1000;
                }
            }
        } catch { /* ignore */ }

        this.duration += evt.result.duration;

        await this.transcriberOutputQueue.put(
            createWsDataPacket({
                data: { type: "transcript", content: text },
                metaInfo: this.metaInfo ?? {},
            })
        );
    }

    private async canceledHandler(evt: sdk.SpeechRecognitionCanceledEventArgs): Promise<void> {
        logger.info(`Canceled event received: ${evt} | run_id - ${this.runId}`);
    }

    private async sessionStartedHandler(_evt: sdk.SessionEventArgs): Promise<void> {
        logger.info(`Session start event received | run_id - ${this.runId}`);
        this.startTime = Date.now() / 1000;
    }

    private async sessionStoppedHandler(_evt: sdk.SessionEventArgs): Promise<void> {
        logger.info(`Session stop event received | run_id - ${this.runId}`);
        this.endTime = Date.now() / 1000;
        if (this.metaInfo) {
            this.metaInfo["transcriber_duration"] = this.endTime - (this.startTime ?? this.endTime);
        }
        await this.transcriberOutputQueue.put(
            createWsDataPacket({
                data: "transcriber_connection_closed",
                metaInfo: this.metaInfo ?? {},
            })
        );
    }

    // ------------------------------------------------------------------
    // Sync cleanup
    // ------------------------------------------------------------------

    private syncCleanup(): void {
        try {
            logger.info("Cleaning up azure connections");

            if (this.pushStream) {
                this.pushStream.close();
                this.pushStream = null;
            }

            if (this.recognizer) {
                try {
                    this.recognizer.stopContinuousRecognitionAsync(
                        () => logger.info("Azure recognition stopped"),
                        (err) => logger.error(`Error stopping recognition: ${err}`)
                    );
                } finally {
                    this.recognizer = null;
                }
            }

            if (!this.endTime) this.endTime = Date.now() / 1000;

            logger.info("Connections to azure have been successfully closed");
            if (this.startTime) {
                logger.info(
                    `Time duration as per azure - ${this.duration} | ` +
                    `Time duration as per self calculation - ${this.endTime - this.startTime}`
                );
            }
        } catch (e) {
            logger.error(`Error occurred while cleaning up: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Toggle / cleanup
    // ------------------------------------------------------------------

    async toggleConnection(): Promise<void> {
        this.connectionOn = false;
        if (this.sendAudioTask) {
            try { await this.sendAudioTask; } catch { /* ignore */ }
            this.sendAudioTask = null;
        }
        this.syncCleanup();
    }

    override async cleanup(): Promise<void> {
        logger.info("Cleaning up Azure transcriber resources");
        this.syncCleanup();
        logger.info("Azure transcriber cleanup complete");
    }

    // ------------------------------------------------------------------
    // Run
    // ------------------------------------------------------------------

    async run(): Promise<void> {
        try {
            await this.initializeConnection();
            this.sendAudioTask = this.sendAudioToTranscriber();
            await this.sendAudioTask;
        } catch (e) {
            logger.error(`Error received in run method: ${e}`);
        }
    }

    getMetaInfo(): Record<string, unknown> {
        return this.metaInfo ?? {};
    }
}