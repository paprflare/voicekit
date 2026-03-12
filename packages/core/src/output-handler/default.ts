import { configureLogger } from "../helper/logger";
import { v4 as uuidv4 } from "uuid";
import type { MarkEventMetaData } from "../helper/markEvent.metadata";

const logger = configureLogger("defaultOutputHandler");

interface MetaInfo {
    type: "audio" | "text" | string;
    sequence_id: number;
    message_category?: string;
    is_first_chunk?: boolean;
    end_of_llm_stream?: boolean;
    end_of_synthesizer_stream?: boolean;
    text_synthesized?: string;
    mark_id?: string;
    is_final_chunk_of_entire_response?: boolean;
}

interface Packet {
    data: Buffer | string;
    meta_info: MetaInfo;
}

interface WebSocket {
    send_json(data: unknown): Promise<void>;
    send_text(data: string): Promise<void>;
}

export class DefaultOutputHandler {
    private websocket: WebSocket | null;
    isInterruptionTaskOn = false;
    private queue: unknown;
    protected ioProvider: string;
    private isChunkingSupported = true;
    private isLastHangupChunkSent = false;
    private isWebBasedCall: boolean;
    private markEventMetaData: MarkEventMetaData | null;
    private welcomeMessageSentTs: number | null = null;
    private _closed = false;

    constructor(opts: {
        ioProvider?: string;
        websocket?: WebSocket | null;
        queue?: unknown;
        isWebBasedCall?: boolean;
        markEventMetaData?: MarkEventMetaData | null;
    } = {}) {
        this.ioProvider = opts.ioProvider ?? "default";
        this.websocket = opts.websocket ?? null;
        this.queue = opts.queue ?? null;
        this.isWebBasedCall = opts.isWebBasedCall ?? false;
        this.markEventMetaData = opts.markEventMetaData ?? null;
    }

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------

    close(): void {
        this._closed = true;
    }

    isClosed(): boolean {
        return this._closed;
    }

    processInChunks(yieldChunks = false): boolean {
        return this.isChunkingSupported && yieldChunks;
    }

    getProvider(): string {
        return this.ioProvider;
    }

    setHangupSent(): void {
        this.isLastHangupChunkSent = true;
    }

    hangupSent(): boolean {
        return this.isLastHangupChunkSent;
    }

    getWelcomeMessageSentTs(): number | null {
        return this.welcomeMessageSentTs;
    }

    requiresCustomVoicemailDetection(): boolean {
        return true;
    }

    // ------------------------------------------------------------------
    // Send helpers
    // ------------------------------------------------------------------

    async sendInitAcknowledgement(): Promise<void> {
        if (this._closed) return;
        try {
            logger.info("Sending ack event");
            await this.websocket!.send_text(JSON.stringify({ type: "ack" }));
        } catch (e) {
            logger.info(`WebSocket closed during init ack: ${e}`);
            this._closed = true;
        }
    }

    async handleInterruption(): Promise<void> {
        if (this._closed) return;
        try {
            await this.websocket!.send_json({ data: null, type: "clear" });
            this.markEventMetaData?.clearData();
        } catch (e) {
            logger.info(`WebSocket closed during interruption: ${e}`);
            this._closed = true;
        }
    }

    async handle(packet: Packet): Promise<void> {
        if (this._closed) return;
        try {
            logger.info("Packet received:");
            const { meta_info: metaInfo } = packet;

            if (metaInfo.type !== "audio" && metaInfo.type !== "text") {
                logger.error("Other modalities are not implemented yet");
                return;
            }

            let data: string;

            if (metaInfo.type === "audio") {
                logger.info("Sending audio");
                data = Buffer.isBuffer(packet.data)
                    ? packet.data.toString("base64")
                    : Buffer.from(packet.data as string).toString("base64");
            } else {
                logger.info(`Sending text response ${packet.data}`);
                data = packet.data as string;
            }

            // Pre-mark message (audio only)
            if (metaInfo.type === "audio") {
                const preMarkId = uuidv4();
                this.markEventMetaData?.updateData(preMarkId, { type: "pre_mark_message" });
                await this.websocket!.send_text(
                    JSON.stringify({ type: "mark", name: preMarkId })
                );
            }

            logger.info(`Sending to the frontend ${data.length}`);

            if (
                metaInfo.message_category === "agent_welcome_message" &&
                !this.welcomeMessageSentTs
            ) {
                this.welcomeMessageSentTs = Date.now();
            }

            await this.websocket!.send_json({ data, type: metaInfo.type });

            // Post-mark message (audio only)
            if (metaInfo.type === "audio") {
                const markEventData = {
                    text_synthesized:
                        metaInfo.sequence_id === -1
                            ? ""
                            : (metaInfo.text_synthesized ?? ""),
                    type: metaInfo.message_category ?? "",
                    is_first_chunk: metaInfo.is_first_chunk ?? false,
                    is_final_chunk:
                        (metaInfo.end_of_llm_stream ?? false) &&
                        (metaInfo.end_of_synthesizer_stream ?? false),
                    sequence_id: metaInfo.sequence_id,
                };

                const markId =
                    metaInfo.mark_id && metaInfo.mark_id !== ""
                        ? metaInfo.mark_id
                        : uuidv4();

                this.markEventMetaData?.updateData(markId, markEventData);
                await this.websocket!.send_text(
                    JSON.stringify({ type: "mark", name: markId })
                );
            }
        } catch (e) {
            this._closed = true;
            logger.debug(`WebSocket send failed (client disconnected): ${e}`);
        }
    }
}