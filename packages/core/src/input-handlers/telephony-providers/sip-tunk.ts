/**
 * Asterisk WebSocket (chan_websocket) input handler for sip-trunk provider.
 * BINARY = ulaw audio; TEXT = control events (JSON or plain text).
 * Ref: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
 */

import { configureLogger } from "../../helper/logger";
import { TelephonyInputHandler } from "../telephony";
import { createWsDataPacket } from "../../helper/utils";
import WebSocket from "ws";
import type { MarkEventMetaData } from "../../helper/markEvent.metadata";
import type { ObservableVariable } from "../../helper/observable.variable";

const logger = configureLogger("sipTrunkInputHandler");

const ASTERISK_ULAW_OPTIMAL_FRAME_SIZE = 160;

// ============================================================
// Control message parser
// ============================================================

function parseAsteriskControlMessage(text: string): Record<string, string> {
    const trimmed = (text ?? "").trim();
    if (!trimmed) return {};

    const result: Record<string, string> = {};

    // Try JSON first
    try {
        const obj = JSON.parse(trimmed) as unknown;
        if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
            Object.assign(result, obj as Record<string, string>);
        }
    } catch {
        // Plain text: "KEY:value" or "KEY value ..."
        for (const part of trimmed.split(/\s+/)) {
            if (part.includes(":")) {
                const [k, v] = part.split(":", 2) as [string, string];
                result[k.trim().toLowerCase()] = v.trim();
            }
        }
        if (trimmed.includes(" ")) {
            const first = trimmed.split(/\s+/)[0]!;
            if (!(first in result)) result["event"] = first;
        }
    }

    const event = (
        (result["event"] ?? result["command"] ?? "") as string
    )
        .toUpperCase()
        .replace(/\s+/g, "_");

    if (event) result["event"] = event;
    return result;
}

// ============================================================
// Types
// ============================================================

interface AsyncQueue<T = unknown> {
    put_nowait(item: T): void;
    put(item: T): Promise<void>;
    get(): Promise<T>;
}

interface Queues {
    transcriber: AsyncQueue;
    llm: AsyncQueue;
    dtmf?: AsyncQueue;
    [key: string]: AsyncQueue | undefined;
}

interface ObservableVariables {
    [key: string]: ObservableVariable<boolean> | undefined;
}

interface MediaStartData {
    channel_id?: string;
    connection_id?: string;
    format?: string;
    ptime?: number | string;
    optimal_frame_size?: number | string;
}

interface OutputHandlerRef {
    queue_full: boolean;
    drainLocalQueue(): Promise<void>;
}

// ============================================================
// SipTrunkInputHandler
// ============================================================

export class SipTrunkInputHandler extends TelephonyInputHandler {
    private agentConfig: Record<string, unknown>;
    private optimalFrameSize: number;
    private mediaXoff = false;
    mediaStarted = false;
    private channelId: string | null = null;
    private connectionId: string | null = null;
    private ptime = 20;
    private expectedFormat: string;
    outputHandlerRef: OutputHandlerRef | null = null;

    constructor(opts: {
        queues: Queues;
        websocket?: WebSocket | null;
        inputTypes?: Record<string, unknown>;
        markEventMetaData?: MarkEventMetaData | null;
        turnBasedConversation?: boolean;
        isWelcomeMessagePlayed?: boolean;
        observableVariables?: ObservableVariables;
        asteriskMediaStart?: MediaStartData | null;
        agentConfig?: Record<string, unknown>;
        wsContextData?: Record<string, unknown> | null;
    }) {
        super({
            queues: opts.queues,
            websocket: opts.websocket,
            inputTypes: opts.inputTypes,
            markEventMetaData: opts.markEventMetaData,
            turnBasedConversation: opts.turnBasedConversation,
            isWelcomeMessagePlayed: opts.isWelcomeMessagePlayed,
            observableVariables: opts.observableVariables,
        });

        this.ioProvider = "sip-trunk";
        this.agentConfig = opts.agentConfig ?? {};
        this.optimalFrameSize = ASTERISK_ULAW_OPTIMAL_FRAME_SIZE;

        const inputConfig = this.getInputConfig();
        const fmt = (
            (inputConfig["audio_format"] as string) ||
            (inputConfig["format"] as string) ||
            "ulaw"
        ).toLowerCase();

        if (!["ulaw", "mulaw"].includes(fmt)) {
            logger.warn(`sip-trunk input expects ulaw; got ${fmt}, using ulaw`);
            this.expectedFormat = "ulaw";
        } else {
            this.expectedFormat = fmt;
        }

        // Resolve MEDIA_START data from wsContextData or asteriskMediaStart
        const mediaStartData =
            (opts.wsContextData?.["media_start_data"] as MediaStartData | undefined) ??
            opts.asteriskMediaStart ??
            null;

        if (mediaStartData) {
            this.initializeFromMediaStart(mediaStartData);
        }
    }

    // ------------------------------------------------------------------
    // Config helpers
    // ------------------------------------------------------------------

    private getInputConfig(): Record<string, unknown> {
        try {
            const tasks = (this.agentConfig["tasks"] as Record<string, unknown>[]) ?? [];
            if (tasks.length && typeof tasks[0] === "object") {
                return (
                    ((tasks[0]!["tools_config"] as Record<string, unknown>)?.["input"] as Record<string, unknown>) ?? {}
                );
            }
        } catch { /* ignore */ }
        return {};
    }

    private initializeFromMediaStart(data: MediaStartData): void {
        this.channelId = data.channel_id ?? null;
        this.connectionId = data.connection_id ?? this.channelId;
        this.ptime = parseInt(String(data.ptime ?? 20), 10) || 20;
        this.streamSid = this.connectionId ?? this.channelId;
        this.callSid =
            this.channelId?.includes("_")
                ? this.channelId.split("_")[0]!
                : this.channelId;

        const opt = data.optimal_frame_size;
        if (opt != null) {
            const parsed = parseInt(String(opt), 10);
            if (!isNaN(parsed)) this.optimalFrameSize = parsed;
        }

        this.mediaStarted = true;
        logger.info(
            `Initialized from MEDIA_START - channel_id=${this.channelId}, format=${data.format ?? "ulaw"}, ptime=${this.ptime}ms`
        );
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    override async callStart(packet: Record<string, unknown>): Promise<void> {
        this.initializeFromMediaStart(packet as MediaStartData);
    }

    override  async disconnectStream(): Promise<void> {
        try {
            if (this.websocket && this.channelId) {
                const ws = this.websocket as WebSocket & { readyState: number };
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send("HANGUP");
                    logger.info(`Sent HANGUP for channel ${this.channelId}`);
                }
            }
        } catch (e) {
            logger.error(`Error sending HANGUP: ${e}`);
        }
    }

    override  async stopHandler(): Promise<void> {
        logger.info(`Stopping sip-trunk handler for channel ${this.channelId}`);
        this.running = false;
        await this.disconnectStream();
        await new Promise((resolve) => setTimeout(resolve, 500));
        try {
            this.websocket?.close();
        } catch (e) {
            logger.info(`Error closing WebSocket: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Control message handler
    // ------------------------------------------------------------------

    private async handleControlMessage(text: string): Promise<void> {
        const parsed = parseAsteriskControlMessage(text);
        const event = parsed["event"] ?? "";

        if (event === "MEDIA_START") {
            await this.callStart(parsed as Record<string, unknown>);
            return;
        }

        if (event === "DTMF_END") {
            const digit = parsed["digit"] ?? "";
            if (digit && this.isDtmfActive) {
                (this.queues as unknown as Queues)["dtmf"]?.put_nowait(digit);
            }
            return;
        }

        if (event === "MEDIA_XOFF") {
            this.mediaXoff = true;
            if (this.outputHandlerRef) this.outputHandlerRef.queue_full = true;
            logger.debug(`MEDIA_XOFF for channel ${this.channelId}`);
            return;
        }

        if (event === "MEDIA_XON") {
            this.mediaXoff = false;
            if (this.outputHandlerRef) {
                this.outputHandlerRef.queue_full = false;
                this.outputHandlerRef
                    .drainLocalQueue()
                    .catch((e) => logger.error(`sip-trunk drain_local_queue failed: ${e}`));
            }
            logger.debug(`MEDIA_XON for channel ${this.channelId}`);
            return;
        }

        if (event === "STATUS" || event.includes("MEDIA_BUFFERING_COMPLETED")) {
            logger.debug(`Asterisk control: ${event}`);
            return;
        }

        if (event === "QUEUE_DRAINED" || event.includes("QUEUE_DRAINED")) {
            // Do NOT use QUEUE_DRAINED to clear is_audio_being_played or process marks.
            // Asterisk sends QUEUE_DRAINED when it has accepted the bulk, not when caller has heard it.
            // Rely on output handler's duration-based fallback instead.
            logger.debug(`QUEUE_DRAINED for channel ${this.channelId} (playback-done handled by fallback)`);
            return;
        }

        if (event || Object.keys(parsed).length) {
            logger.debug(`Asterisk control: ${text} -> event=${event}`);
        }
    }

    // ------------------------------------------------------------------
    // Listen loop
    // ------------------------------------------------------------------

    protected override listen(): Promise<void> {
        return new Promise<void>((resolve) => {
            const ws = this.websocket! as WebSocket;
            const buffer: Buffer[] = [];
            // ~80ms per batch (4 × 20ms frames) for stable Deepgram transcripts
            const chunksPerBatch = Math.max(2, Math.floor(80 / (this.ptime || 20)));
            let messageCount = 0;

            const buildMetaInfo = (): Record<string, unknown> => ({
                io: this.ioProvider,
                call_sid: this.callSid,
                stream_sid: this.streamSid,
                sequence: (this.inputTypes as Record<string, unknown>)?.["audio"] ?? 0,
                format: this.expectedFormat,
            });

            const flushBuffer = async () => {
                if (!buffer.length) return;
                const merged = Buffer.concat(buffer);
                buffer.length = 0;
                messageCount = 0;
                if (merged.length) {
                    await this.ingestAudio(merged, buildMetaInfo());
                }
            };

            const sendEos = () => {
                const wsDataPacket = createWsDataPacket({
                    data: null,
                    metaInfo: {
                        io: this.ioProvider,
                        eos: true,
                        sequence: (this.inputTypes as Record<string, unknown>)?.["audio"] ?? 0,
                    },
                });
                (this.queues as unknown as Queues)["transcriber"].put_nowait(wsDataPacket);
                logger.info(`sip-trunk WebSocket closed for channel ${this.channelId}`);
            };

            const onMessage = async (data: WebSocket.RawData, isBinary: boolean) => {
                if (!this.running) return;
                try {
                    if (isBinary) {
                        // Binary = ulaw audio
                        const chunk = Buffer.isBuffer(data) ? data : Buffer.from(data as ArrayBuffer);
                        if (!chunk.length) return;
                        buffer.push(chunk);
                        messageCount += 1;
                        if (messageCount >= chunksPerBatch) {
                            await flushBuffer();
                        }
                    } else {
                        // Text = Asterisk control event
                        await this.handleControlMessage(data.toString());
                    }
                } catch (e) {
                    logger.error(`Error in sip-trunk listen: ${e}`);
                    console.error(e);
                }
            };

            const onClose = async (code: number) => {
                const normalCodes = new Set([1000, 1001, 1006]);
                if (normalCodes.has(code)) {
                    logger.info("WebSocket disconnected normally");
                } else {
                    logger.error(`WebSocket disconnected: code=${code}`);
                }
                await flushBuffer().catch(() => { });
                sendEos();
                cleanup();
                resolve();
            };

            const onError = async (err: Error) => {
                logger.error(`Runtime error in sip-trunk: ${err}`);
                await flushBuffer().catch(() => { });
                sendEos();
                cleanup();
                resolve();
            };

            const cleanup = () => {
                ws.off("message", onMessage);
                ws.off("close", onClose);
                ws.off("error", onError);
            };

            ws.on("message", onMessage);
            ws.once("close", onClose);
            ws.once("error", onError);
        });
    }

    override async handle(): Promise<void> {
        if (!this.websocketListenTask) {
            this.websocketListenTask = this.listen();
        }
    }
}