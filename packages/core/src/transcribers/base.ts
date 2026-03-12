import { configureLogger } from "../helper/logger";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("baseTranscriber");

export abstract class BaseTranscriber {
    protected inputQueue: unknown | null;
    protected connectionOn = true;
    protected calleeSpeaking = false;
    protected callerSpeaking = false;
    protected metaInfo: Record<string, unknown> | null = null;
    protected transcriptionStartTime = 0;
    protected lastVocalFrameTime: number | null = null;
    protected previousRequestId: string | null = null;
    protected currentRequestId: string | null = null;
    protected connectionTime: number | null = null;
    protected turnLatencies: Record<string, unknown>[] = [];

    constructor(inputQueue: unknown | null = null) {
        this.inputQueue = inputQueue;
    }

    // ------------------------------------------------------------------
    // Meta info
    // ------------------------------------------------------------------

    updateMetaInfo(): void {
        if (!this.metaInfo) return;
        this.metaInfo["request_id"] = this.currentRequestId ?? null;
        this.metaInfo["previous_request_id"] = this.previousRequestId;
        this.metaInfo["origin"] = "transcriber";
    }

    static generateRequestId(): string {
        return uuidv4();
    }

    // ------------------------------------------------------------------
    // Transcription lifecycle
    // ------------------------------------------------------------------

    async signalTranscriptionBegin(msg: { duration: number }): Promise<boolean> {
        let sendBeginPacket = false;
        if (this.metaInfo) {
            this.metaInfo["request_id"] = this.currentRequestId;
        }

        if (!this.calleeSpeaking) {
            this.calleeSpeaking = true;
            logger.debug("Making callee speaking true");
            this.transcriptionStartTime = Date.now() / 1000 - msg.duration;
            sendBeginPacket = true;
        }

        return sendBeginPacket;
    }

    async logLatencyInfo(): Promise<void> {
        const transcriptionCompletionTime = Date.now() / 1000;
        if (this.lastVocalFrameTime) {
            logger.info(
                `################ Time latency: For request ${this.metaInfo?.["request_id"]}, ` +
                `user started speaking at ${this.transcriptionStartTime}, ` +
                `last audio frame received at ${this.lastVocalFrameTime}, ` +
                `transcription_completed_at ${transcriptionCompletionTime}, ` +
                `overall latency ${transcriptionCompletionTime - this.lastVocalFrameTime}`
            );
        } else {
            logger.info(
                `No confidence for the last vocal timeframe. ` +
                `Over transcription time ${transcriptionCompletionTime - this.transcriptionStartTime}`
            );
        }
    }

    // ------------------------------------------------------------------
    // WebSocket close helper
    // ------------------------------------------------------------------

    async _close(
        ws: { send: (data: string) => Promise<void> } | null,
        data: Record<string, unknown>
    ): Promise<void> {
        if (!ws) {
            logger.warn("Transcriber websocket already closed, skipping close message");
            return;
        }
        try {
            await ws.send(JSON.stringify(data));
        } catch (e) {
            logger.error(`Error while closing transcriber stream: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Cleanup (override in subclasses)
    // ------------------------------------------------------------------

    async cleanup(): Promise<void> { }

    // ------------------------------------------------------------------
    // Latency helpers
    // ------------------------------------------------------------------

    calculateInterimToFinalLatencies(
        interimDetails: { received_at?: number }[]
    ): [number | null, number | null] {
        if (!interimDetails.length) return [null, null];

        const now = Date.now() / 1000;
        const firstReceivedAt = interimDetails[0]?.received_at;
        const lastReceivedAt = interimDetails[interimDetails.length - 1]?.received_at;

        const firstInterimToFinalMs = firstReceivedAt != null
            ? Math.round((now - firstReceivedAt) * 1000 * 100) / 100
            : null;
        const lastInterimToFinalMs = lastReceivedAt != null
            ? Math.round((now - lastReceivedAt) * 1000 * 100) / 100
            : null;

        return [firstInterimToFinalMs, lastInterimToFinalMs];
    }
}