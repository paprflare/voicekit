import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket, ulaw2lin, type WsDataPacket } from "../helper/utils";
import { BaseTranscriber } from "./base";

const logger = configureLogger("sarvamTranscriber");

export class SarvamTranscriber extends BaseTranscriber {
    private telephonyProvider: string;
    private model: string;
    private language: string;
    private targetLanguage: string | null;
    private stream: boolean;
    private encoding: string;
    private samplingRate: number;
    private inputSamplingRate: number;
    private audioFrameDuration: number;
    private highVadSensitivity: boolean;
    private vadSignals: boolean;
    private disableSdk: boolean;
    private apiKey: string;
    private apiHost: string;
    private apiUrl: string;
    private wsUrl: string;

    private transcriberOutputQueue: { put: (data: unknown) => Promise<void> } | null;
    private transcriptionTask: Promise<void> | null = null;
    private senderTask: Promise<void> | null = null;
    private heartbeatTask: Promise<void> | null = null;

    private audioSubmitted = false;
    private audioSubmissionTime: number | null = null;
    private numFrames = 0;
    private connectionStartTime: number | null = null;
    private audioCursor = 0.0;

    private finalTranscript = "";
    private websocketConnection: WebSocket | null = null;
    private connectionAuthenticated = false;
    override metaInfo: Record<string, unknown> = {};

    private currentTurnStartTime: number | null = null;
    private currentTurnId: string | null = null;
    private firstResultLatencyMs: number | null = null;
    private totalStreamDurationMs: number | null = null;
    private lastVocalFrameTimestamp: number | null = null;
    private turnCounter = 0;
    private turnFirstResultLatency: number | null = null;

    private isTranscriptSentForProcessing = false;
    private currMessage = "";
    private finalizedTranscript = "";
    private interruptionSignalled = false;

    // Abort controller for cancellable tasks
    private abortController = new AbortController();

    constructor(opts: {
        telephonyProvider: string;
        inputQueue?: { get: () => Promise<unknown> } | null;
        model?: string;
        stream?: boolean;
        language?: string;
        targetLanguage?: string | null;
        encoding?: string;
        samplingRate?: string;
        outputQueue?: { put: (data: unknown) => Promise<void> } | null;
        highVadSensitivity?: boolean;
        vadSignals?: boolean;
        disableSdk?: boolean;
        transcriberKey?: string;
        [key: string]: unknown;
    }) {
        super(opts.inputQueue ?? null);

        this.telephonyProvider = opts.telephonyProvider;
        this.model = opts.model ?? "saarika:v2.5";
        this.language = opts.language ?? "en-IN";
        this.targetLanguage = opts.targetLanguage ?? null;
        this.stream = opts.stream ?? true;
        this.encoding = opts.encoding ?? "linear16";
        this.samplingRate = parseInt(opts.samplingRate ?? "16000", 10);
        this.inputSamplingRate = this.samplingRate;
        this.audioFrameDuration = 0.2;
        this.highVadSensitivity = opts.highVadSensitivity ?? true;
        this.vadSignals = opts.vadSignals ?? true;
        this.disableSdk = opts.disableSdk ?? false;

        this.apiKey = (opts.transcriberKey as string | undefined) ?? process.env.SARVAM_API_KEY!;
        this.apiHost = process.env.SARVAM_HOST ?? "api.sarvam.ai";
        this.transcriberOutputQueue = opts.outputQueue ?? null;

        this.apiUrl = "";
        this.wsUrl = "";
        this.setEndpoints();
        this.configureAudioParams();
    }

    // ------------------------------------------------------------------
    // Config
    // ------------------------------------------------------------------

    private configureAudioParams(): void {
        if (this.telephonyProvider === "plivo") {
            this.encoding = "linear16";
            this.inputSamplingRate = 8000;
            this.samplingRate = 16000;
            this.audioFrameDuration = 0.2;
        } else if (this.telephonyProvider === "twilio") {
            this.encoding = "mulaw";
            this.inputSamplingRate = 8000;
            this.samplingRate = 16000;
            this.audioFrameDuration = 0.2;
        } else {
            this.encoding = this.encoding || "linear16";
            this.inputSamplingRate = this.samplingRate;
            this.audioFrameDuration = 0.2;
        }
    }

    private setEndpoints(): void {
        const params: Record<string, string> = { model: this.model };
        let wsBase: string;

        if (this.model.startsWith("saaras") && this.model !== "saaras:v3") {
            this.apiUrl = `https://${this.apiHost}/speech-to-text-translate`;
            wsBase = `wss://${this.apiHost}/speech-to-text-translate/ws`;
        } else {
            this.apiUrl = `https://${this.apiHost}/speech-to-text`;
            wsBase = `wss://${this.apiHost}/speech-to-text/ws`;
            params["language-code"] = this.language;
        }

        if (this.model === "saaras:v3") params["mode"] = "transcribe";
        if (this.highVadSensitivity) params["high_vad_sensitivity"] = "true";
        if (this.vadSignals) params["vad_signals"] = "true";
        if (this.targetLanguage) params["target_language"] = this.targetLanguage;

        const qs = Object.entries(params).map(([k, v]) => `${k}=${v}`).join("&");
        this.wsUrl = `${wsBase}?${qs}`;
    }

    // ------------------------------------------------------------------
    // Audio helpers
    // ------------------------------------------------------------------

    private convertAudioToWav(audioData: Buffer | string): Buffer | null {
        try {
            let audioBytes: Buffer =
                typeof audioData === "string"
                    ? Buffer.from(audioData, "base64")
                    : audioData;

            // μ-law → PCM
            if (this.encoding === "mulaw") {
                audioBytes = ulaw2lin(audioBytes);
            }

            // Resample if needed
            if (this.inputSamplingRate !== this.samplingRate) {
                audioBytes = this.resamplePcm(audioBytes, this.inputSamplingRate, this.samplingRate);
            }

            // Build WAV header manually (PCM, mono, 16-bit)
            const numSamples = audioBytes.length / 2;
            const header = Buffer.alloc(44);
            header.write("RIFF", 0);
            header.writeUInt32LE(36 + audioBytes.length, 4);
            header.write("WAVE", 8);
            header.write("fmt ", 12);
            header.writeUInt32LE(16, 16);           // PCM chunk size
            header.writeUInt16LE(1, 20);            // PCM format
            header.writeUInt16LE(1, 22);            // mono
            header.writeUInt32LE(this.samplingRate, 24);
            header.writeUInt32LE(this.samplingRate * 2, 28); // byte rate
            header.writeUInt16LE(2, 32);            // block align
            header.writeUInt16LE(16, 34);           // bits per sample
            header.write("data", 36);
            header.writeUInt32LE(audioBytes.length, 40);

            return Buffer.concat([header, audioBytes]);
        } catch (e) {
            logger.error(`WAV conversion error for language ${this.language}: ${e}`);
            logger.warn("Skipping audio frame - this may cause missing transcripts");
            return null;
        }
    }

    // Simple linear interpolation resample (replaces scipy.signal.resample_poly)
    private resamplePcm(pcm: Buffer, inRate: number, outRate: number): Buffer {
        if (inRate === outRate) return pcm;
        const samples = new Int16Array(pcm.buffer, pcm.byteOffset, pcm.length / 2);
        const ratio = outRate / inRate;
        const outLen = Math.round(samples.length * ratio);
        const out = new Int16Array(outLen);
        for (let i = 0; i < outLen; i++) {
            const src = i / ratio;
            const lo = Math.floor(src);
            const hi = Math.min(lo + 1, samples.length - 1);
            const frac = src - lo;
            out[i] = Math.round(samples[lo]! * (1 - frac) + samples[hi]! * frac);
        }
        return Buffer.from(out.buffer);
    }

    // ------------------------------------------------------------------
    // HTTP transcription
    // ------------------------------------------------------------------

    private async getHttpTranscription(audioData: Buffer): Promise<WsDataPacket> {
        const wavData = this.convertAudioToWav(audioData);
        if (!wavData) return createWsDataPacket({ data: "", metaInfo: this.metaInfo });

        try {
            const formData = new FormData();
            formData.append("file", new Blob([Buffer.from(wavData)], { type: "audio/wav" }), "audio.wav");
            formData.append("model", this.model);
            formData.append("language_code", this.language);

            this.currentRequestId = BaseTranscriber.generateRequestId();
            this.metaInfo["request_id"] = this.currentRequestId;
            const startTime = Date.now() / 1000;

            const res = await fetch(this.apiUrl, {
                method: "POST",
                headers: { "api-subscription-key": this.apiKey },
                body: formData,
            });

            const text = await res.text();

            if (res.ok) {
                const data = JSON.parse(text) as Record<string, unknown>;
                const elapsed = Date.now() / 1000 - startTime;
                this.metaInfo["start_time"] = startTime;
                this.metaInfo["transcriber_first_result_latency"] = elapsed;
                this.metaInfo["transcriber_latency"] = elapsed;
                this.metaInfo["first_result_latency_ms"] = Math.round(elapsed * 1000);
                this.metaInfo["transcriber_duration"] = data["duration"] ?? 0;
                const transcript = (data["transcript"] as string) ?? "";
                return createWsDataPacket({ data: transcript, metaInfo: this.metaInfo });
            } else if (res.status === 429) {
                throw new Error("Rate limit exceeded");
            } else {
                return createWsDataPacket({ data: "", metaInfo: this.metaInfo });
            }
        } catch (e) {
            logger.error(`HTTP transcription error: ${e}`);
            throw e;
        }
    }

    // ------------------------------------------------------------------
    // WebSocket connection
    // ------------------------------------------------------------------

    private async sarvamConnect(retries = 3, timeoutMs = 10_000): Promise<WebSocket> {
        let attempt = 0;
        let lastErr: unknown;

        while (attempt < retries) {
            logger.info(`Attempting to connect to Sarvam websocket: ${this.wsUrl}`);
            try {
                const ws = await new Promise<WebSocket>((resolve, reject) => {
                    const timer = setTimeout(() => reject(new Error("Timeout")), timeoutMs);
                    const ws = new WebSocket(this.wsUrl, {
                        headers: { "api-subscription-key": this.apiKey },
                    });
                    ws.once("open", () => { clearTimeout(timer); resolve(ws); });
                    ws.once("error", (err) => { clearTimeout(timer); reject(err); });
                });

                this.websocketConnection = ws;
                this.connectionAuthenticated = true;
                logger.info("Successfully connected to Sarvam websocket");
                return ws;
            } catch (e) {
                const msg = (e as Error).message ?? "";
                if (msg.includes("Timeout")) {
                    throw new ConnectionError("Timeout while connecting to Sarvam websocket");
                }
                if (msg.includes("401") || msg.includes("403")) {
                    throw new ConnectionError(`Sarvam authentication failed: ${e}`);
                }
                if (msg.includes("404")) {
                    throw new ConnectionError(`Sarvam endpoint not found: ${e}`);
                }
                lastErr = e;
                attempt++;
                logger.error(`Sarvam connect attempt ${attempt}/${retries} failed: ${e}`);
                if (attempt < retries) {
                    await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 1000));
                }
            }
        }
        throw new ConnectionError(`Failed to connect to Sarvam after ${retries} attempts: ${lastErr}`);
    }

    // ------------------------------------------------------------------
    // Sender (HTTP batching mode)
    // ------------------------------------------------------------------

    private async sender(): Promise<void> {
        const bufferFlushIntervalMs = 2500;
        let lastFlushTime = Date.now();
        const audioBuffer: Buffer[] = [];
        let consecutiveErrors = 0;

        try {
            while (true) {
                const wsDataPacket = await (this.inputQueue as { get: () => Promise<Record<string, unknown>> }).get();
                if (!wsDataPacket) continue;

                const metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                if (metaInfo["eos"] === true) {
                    if (audioBuffer.length) {
                        const combined = Buffer.concat(audioBuffer);
                        const packet = await this.getHttpTranscription(combined);
                        await this.pushToTranscriberQueue(packet);
                        audioBuffer.length = 0;
                    }
                    break;
                }

                if (!this.audioSubmitted) {
                    this.metaInfo = metaInfo;
                    this.audioSubmitted = true;
                    this.audioSubmissionTime = Date.now() / 1000;
                    this.currentRequestId = BaseTranscriber.generateRequestId();
                    this.metaInfo["request_id"] = this.currentRequestId;
                }

                this.numFrames++;
                const frame = wsDataPacket["data"] as Buffer | undefined;
                if (frame) audioBuffer.push(frame);

                const now = Date.now();
                if (now - lastFlushTime >= bufferFlushIntervalMs && audioBuffer.length) {
                    try {
                        const combined = Buffer.concat(audioBuffer);
                        const packet = await this.getHttpTranscription(combined);
                        await this.pushToTranscriberQueue(packet);
                        lastFlushTime = now;
                        audioBuffer.length = 0;
                        consecutiveErrors = 0;
                    } catch {
                        consecutiveErrors++;
                        if (consecutiveErrors >= 3) break;
                        await new Promise((r) => setTimeout(r, Math.min(Math.pow(2, consecutiveErrors) * 1000, 10_000)));
                    }
                }
            }
        } catch (e) {
            if ((e as Error).message !== "cancelled") throw e;
        }
    }

    // ------------------------------------------------------------------
    // Sender stream (WebSocket mode)
    // ------------------------------------------------------------------

    private async senderStream(ws: WebSocket): Promise<void> {
        try {
            while (true) {
                const wsDataPacket = await (this.inputQueue as { get: () => Promise<Record<string, unknown>> }).get();
                if (!wsDataPacket) continue;

                if (!this.audioSubmitted) {
                    this.metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                    this.audioSubmitted = true;
                    this.audioSubmissionTime = Date.now() / 1000;
                    this.currentRequestId = BaseTranscriber.generateRequestId();
                    this.metaInfo["request_id"] = this.currentRequestId;
                }

                const metaInfo = (wsDataPacket["meta_info"] as Record<string, unknown>) ?? {};
                if (metaInfo["eos"] === true) {
                    try { ws.close(); } catch { /* ignore */ }
                    break;
                }

                this.numFrames++;
                this.audioCursor = this.numFrames * this.audioFrameDuration;

                const audioData = wsDataPacket["data"] as Buffer | undefined;
                if (audioData) {
                    const wavBytes = this.convertAudioToWav(audioData);
                    if (!wavBytes) continue;
                    const audioB64 = wavBytes.toString("base64");
                    ws.send(JSON.stringify({
                        audio: { data: audioB64, encoding: "audio/wav", sample_rate: this.samplingRate },
                    }));
                }
            }
        } catch (e) {
            if ((e as Error).message !== "cancelled") throw e;
        }
    }

    // ------------------------------------------------------------------
    // Receiver async generator
    // ------------------------------------------------------------------

    private async *receiver(ws: WebSocket): AsyncGenerator<WsDataPacket> {
        const messageQueue: (string | Buffer)[] = [];
        const messageResolvers: ((msg: string | Buffer | null) => void)[] = [];
        let closed = false;

        ws.on("message", (raw) => {
            const msg = Buffer.isBuffer(raw) ? raw : raw.toString();
            const resolver = messageResolvers.shift();
            if (resolver) resolver(msg);
            else messageQueue.push(msg);
        });

        ws.once("close", () => {
            closed = true;
            for (const r of messageResolvers) r(null);
        });

        ws.once("error", () => {
            closed = true;
            for (const r of messageResolvers) r(null);
        });

        const nextMessage = (): Promise<string | Buffer | null> =>
            new Promise((resolve) => {
                const pending = messageQueue.shift();
                if (pending !== undefined) resolve(pending);
                else if (closed) resolve(null);
                else messageResolvers.push(resolve);
            });

        try {
            while (true) {
                const raw = await nextMessage();
                if (raw === null) break;

                try {
                    const data = JSON.parse(raw.toString()) as Record<string, unknown>;

                    if (this.connectionStartTime === null) {
                        this.connectionStartTime = Date.now() / 1000 - this.numFrames * this.audioFrameDuration;
                    }

                    // --- Data (transcript) ---
                    if (data["type"] === "data") {
                        const payload = (data["data"] as Record<string, unknown>) ?? {};
                        const transcript = (payload["transcript"] as string) ?? "";
                        const metrics = (payload["metrics"] as Record<string, unknown>) ?? {};

                        if (transcript.trim()) {
                            logger.debug(`Sarvam transcript received: ${transcript.trim().slice(0, 50)}...`);
                            const nowTs = Date.now() / 1000;

                            if (this.firstResultLatencyMs === null && this.audioSubmissionTime !== null) {
                                const latSec = nowTs - this.audioSubmissionTime;
                                this.firstResultLatencyMs = Math.round(latSec * 1000);
                                this.metaInfo["transcriber_first_result_latency"] = latSec;
                                this.metaInfo["transcriber_latency"] = latSec;
                                this.metaInfo["first_result_latency_ms"] = this.firstResultLatencyMs;
                            }

                            if (this.currentTurnStartTime !== null && this.turnFirstResultLatency === null) {
                                const turnLatSec = (performance.now() - this.currentTurnStartTime) / 1000;
                                this.turnFirstResultLatency = Math.round(turnLatSec * 1000);
                                this.metaInfo["transcriber_first_result_latency"] = turnLatSec;
                                this.metaInfo["transcriber_latency"] = turnLatSec;
                            }

                            this.metaInfo["transcriber_duration"] = metrics["audio_duration"] ?? 0;
                            this.lastVocalFrameTimestamp = nowTs;
                            this.metaInfo["last_vocal_frame_timestamp"] = this.lastVocalFrameTimestamp;

                            yield createWsDataPacket({
                                data: { type: "interim_transcript_received", content: transcript.trim() },
                                metaInfo: this.metaInfo,
                            });
                            yield createWsDataPacket({
                                data: { type: "transcript", content: transcript.trim() },
                                metaInfo: this.metaInfo,
                            });
                        }
                    }

                    // --- Events (VAD) ---
                    else if (data["type"] === "events") {
                        const vad = (data["data"] as Record<string, unknown>) ?? {};

                        if (vad["signal_type"] === "START_SPEECH") {
                            logger.debug("Sarvam VAD: speech started");
                            this.currentTurnStartTime = performance.now();
                            this.turnCounter++;
                            this.currentTurnId = `turn_${this.turnCounter}`;
                            this.turnFirstResultLatency = null;
                            yield createWsDataPacket({ data: "speech_started", metaInfo: this.metaInfo });

                        } else if (vad["signal_type"] === "END_SPEECH") {
                            logger.debug("Sarvam VAD: speech ended");
                            const now = Date.now() / 1000;
                            this.lastVocalFrameTimestamp = now;
                            this.metaInfo["last_vocal_frame_timestamp"] = now;

                            if (this.currentTurnStartTime !== null) {
                                const totalSec = (performance.now() - this.currentTurnStartTime) / 1000;
                                const totalMs = Math.round(totalSec * 1000);
                                this.metaInfo["transcriber_total_stream_duration"] = totalSec;
                                this.metaInfo["transcriber_latency"] = totalSec;

                                this.turnLatencies.push({
                                    turn_id: this.currentTurnId,
                                    sequence_id: this.currentTurnId,
                                    first_result_latency_ms: this.turnFirstResultLatency,
                                    total_stream_duration_ms: totalMs,
                                });
                                this.metaInfo["turn_latencies"] = this.turnLatencies;
                                this.currentTurnStartTime = null;
                                this.currentTurnId = null;
                            }

                            if (this.finalTranscript) {
                                yield createWsDataPacket({
                                    data: { type: "transcript", content: this.finalTranscript },
                                    metaInfo: this.metaInfo,
                                });
                                this.finalTranscript = "";
                            }

                            yield createWsDataPacket({ data: "speech_ended", metaInfo: this.metaInfo });
                        }
                    }

                    // --- Connection closed ---
                    else if (data["type"] === "connection_closed") {
                        this.metaInfo["transcriber_duration"] = data["duration"] ?? 0;
                        yield createWsDataPacket({ data: "transcriber_connection_closed", metaInfo: this.metaInfo });
                        return;
                    }
                } catch (e) {
                    console.error(e);
                }
            }
        } catch (e) {
            console.error(e);
        }
    }

    // ------------------------------------------------------------------
    // Heartbeat
    // ------------------------------------------------------------------

    private async sendHeartbeat(ws: WebSocket, intervalSec = 10): Promise<void> {
        try {
            while (true) {
                await new Promise((r) => setTimeout(r, intervalSec * 1000));
                if (ws.readyState !== WebSocket.OPEN) break;
                try {
                    ws.ping();
                } catch { break; }
            }
        } catch { /* cancelled */ }
    }

    // ------------------------------------------------------------------
    // Queue
    // ------------------------------------------------------------------

    async pushToTranscriberQueue(dataPacket: unknown): Promise<void> {
        if (this.transcriberOutputQueue) {
            await this.transcriberOutputQueue.put(dataPacket);
        }
    }

    // ------------------------------------------------------------------
    // Toggle / cleanup
    // ------------------------------------------------------------------

    async toggleConnection(): Promise<void> {
        this.connectionOn = false;
        this.abortController.abort();
        if (this.websocketConnection) {
            try { this.websocketConnection.close(); } catch { /* ignore */ }
            this.websocketConnection = null;
            this.connectionAuthenticated = false;
        }
    }

    override async cleanup(): Promise<void> {
        logger.info("Cleaning up Sarvam transcriber resources");
        this.abortController.abort();

        if (this.websocketConnection) {
            try { this.websocketConnection.close(); } catch (e) {
                logger.error(`Error closing Sarvam websocket: ${e}`);
            }
            this.websocketConnection = null;
            this.connectionAuthenticated = false;
        }
        logger.info("Sarvam transcriber cleanup complete");
    }

    // ------------------------------------------------------------------
    // Transcribe (main entry)
    // ------------------------------------------------------------------

    private async transcribe(): Promise<void> {
        const start = performance.now();
        let sarvamWs: WebSocket;

        try {
            sarvamWs = await this.sarvamConnect();
        } catch (e) {
            logger.error(`Failed to connect: ${e}`);
            await this.toggleConnection();
            return;
        }

        if (!this.connectionTime) {
            this.connectionTime = Math.round(performance.now() - start);
        }

        try {
            if (this.stream) {
                this.senderTask = this.senderStream(sarvamWs);
                this.heartbeatTask = this.sendHeartbeat(sarvamWs);

                for await (const message of this.receiver(sarvamWs)) {
                    if (this.connectionOn) {
                        await this.pushToTranscriberQueue(message);
                    } else {
                        try {
                            sarvamWs.send(JSON.stringify({ type: "CloseStream" }));
                            sarvamWs.close();
                        } catch { /* ignore */ }
                        break;
                    }
                }

                await Promise.allSettled([this.senderTask, this.heartbeatTask]);
            } else {
                this.senderTask = this.sender();
                await this.senderTask;
            }
        } catch (e) {
            console.error(e);
        } finally {
            if (this.audioSubmissionTime !== null) {
                this.totalStreamDurationMs = Math.round((Date.now() / 1000 - this.audioSubmissionTime) * 1000);
                this.metaInfo["total_stream_duration_ms"] = this.totalStreamDurationMs;
            }

            try {
                await this.pushToTranscriberQueue(
                    createWsDataPacket({ data: "transcriber_connection_closed", metaInfo: this.metaInfo })
                );
            } catch (e) {
                console.error(e);
            }

            if (this.websocketConnection) {
                try { this.websocketConnection.close(); } catch { /* ignore */ }
                this.websocketConnection = null;
                this.connectionAuthenticated = false;
            }
        }
    }

    async run(): Promise<void> {
        try {
            this.transcriptionTask = this.transcribe();
            await this.transcriptionTask;
        } catch (e) {
            console.error(e);
        }
    }

    getMetaInfo(): Record<string, unknown> {
        return this.metaInfo;
    }
}

class ConnectionError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "ConnectionError";
    }
}