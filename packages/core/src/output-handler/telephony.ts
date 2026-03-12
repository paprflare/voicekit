import { configureLogger } from "../helper/logger";
import { DefaultOutputHandler } from "./default";
import type { MarkEventMetaData } from "../helper/markEvent.metadata";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("telephonyOutputHandler");

interface MetaInfo {
    sequence_id: number;
    stream_sid?: string;
    format?: string;
    message_category?: string;
    cached?: boolean;
    is_first_chunk?: boolean;
    end_of_llm_stream?: boolean;
    end_of_synthesizer_stream?: boolean;
    text_synthesized?: string;
    mark_id?: string;
}

interface Packet {
    data: Buffer | string;
    meta_info: MetaInfo;
}

export class TelephonyOutputHandler extends DefaultOutputHandler {
    protected streamSid: string | null = null;
    protected currentRequestId: string | null = null;
    protected rejectedRequestIds: Set<string> = new Set();

    constructor(opts: {
        ioProvider: string;
        websocket?: unknown;
        markEventMetaData?: MarkEventMetaData | null;
        logDirName?: string | null;
    }) {
        super({
            ioProvider: opts.ioProvider ?? "",
            websocket: opts.websocket as never,
            markEventMetaData: opts.markEventMetaData,
        });
        // markEventMetaData already set by super; reassign for typed access
    }

    // ------------------------------------------------------------------
    // Overrides (no-op stubs, subclasses implement per-provider logic)
    // ------------------------------------------------------------------

    override async handleInterruption(): Promise<void> {
        // No-op for telephony — subclasses override per provider
    }

    async formMediaMessage(
        _audioData: Buffer,
        _audioFormat: string
    ): Promise<Record<string, unknown>> {
        return {};
    }

    async formMarkMessage(
        _markId: string
    ): Promise<Record<string, unknown>> {
        return {};
    }

    // ------------------------------------------------------------------
    // Stream SID
    // ------------------------------------------------------------------

    async setStreamSid(streamId: string): Promise<void> {
        this.streamSid = streamId;
    }

    // ------------------------------------------------------------------
    // Handle
    // ------------------------------------------------------------------

    override async handle(wsDataPacket: Packet): Promise<void> {
        if (this.isClosed()) return;

        try {
            let audioChunk = wsDataPacket.data ?? Buffer.alloc(0);
            const metaInfo = wsDataPacket.meta_info;

            if (!metaInfo) return;

            if (!this.streamSid) {
                this.streamSid = metaInfo.stream_sid ?? null;
            }

            try {
                // Pad single-byte chunk
                if (audioChunk.length === 1) {
                    audioChunk = Buffer.concat([Buffer.from(audioChunk), Buffer.from([0x00])]);
                }

                if (audioChunk.length && this.streamSid && audioChunk.length !== 1) {
                    const isSilence =
                        audioChunk.length === 2 &&
                        audioChunk[0] === 0x00 &&
                        audioChunk[1] === 0x00;

                    if (!isSilence) {
                        let audioFormat = metaInfo.format ?? "wav";

                        // Pre-mark message
                        const preMarkId = uuidv4();
                        (this as unknown as { markEventMetaData: MarkEventMetaData })
                            .markEventMetaData?.updateData(preMarkId, { type: "pre_mark_message" });
                        const preMarkMessage = await this.formMarkMessage(preMarkId);
                        await (this as unknown as { websocket: { send_text: (s: string) => Promise<void> } })
                            .websocket.send_text(JSON.stringify(preMarkMessage));

                        // PCM → WAV fix for welcome message on some providers
                        if (
                            audioFormat === "pcm" &&
                            metaInfo.message_category === "agent_welcome_message" &&
                            ["plivo", "vobiz"].includes(this.ioProvider) &&
                            metaInfo.cached === true
                        ) {
                            audioFormat = "wav";
                        }

                        // Send audio chunk
                        const mediaMessage = await this.formMediaMessage(Buffer.from(audioChunk), audioFormat);
                        await (this as unknown as { websocket: { send_text: (s: string) => Promise<void> } })
                            .websocket.send_text(JSON.stringify(mediaMessage));

                        if (
                            metaInfo.message_category === "agent_welcome_message" &&
                            !this.getWelcomeMessageSentTs()
                        ) {
                            // welcomeMessageSentTs set via parent's handle path — update directly
                            (this as unknown as Record<string, unknown>)["welcomeMessageSentTs"] = Date.now();
                        }

                        logger.info(`Sending media event - ${metaInfo.mark_id}`);
                    }

                    // Post-mark message (always, even for silence)
                    const isMulaw = (metaInfo.format ?? "mulaw") === "mulaw";
                    const duration = isMulaw
                        ? audioChunk.length / 8000
                        : audioChunk.length / 16000;

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
                        duration,
                        sent_ts: Date.now() / 1000,
                    };

                    const markId =
                        metaInfo.mark_id && metaInfo.mark_id !== ""
                            ? metaInfo.mark_id
                            : uuidv4();

                    (this as unknown as { markEventMetaData: MarkEventMetaData })
                        .markEventMetaData?.updateData(markId, markEventData);
                    const markMessage = await this.formMarkMessage(markId);
                    await (this as unknown as { websocket: { send_text: (s: string) => Promise<void> } })
                        .websocket.send_text(JSON.stringify(markMessage));
                } else {
                    logger.info("Not sending");
                }
            } catch (e) {
                this.close();
                logger.debug(`WebSocket send failed (client disconnected): ${e}`);
            }
        } catch (e) {
            this.close();
            logger.debug(`WebSocket handling failed (client disconnected): ${e}`);
        }
    }

    protected override ioProvider: string = this.getProvider()
}