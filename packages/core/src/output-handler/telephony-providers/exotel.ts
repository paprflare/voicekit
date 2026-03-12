import { configureLogger } from "../../helper/logger";
import { TelephonyOutputHandler } from "../telephony";
import type { MarkEventMetaData } from "../../helper/markEvent.metadata";
import { ulaw2lin } from "../../helper/utils";

const logger = configureLogger("exotelOutputHandler");

export class ExotelOutputHandler extends TelephonyOutputHandler {
    constructor(opts: {
        websocket?: unknown;
        markEventMetaData?: MarkEventMetaData | null;
        logDirName?: string | null;
    } = {}) {
        super({
            ioProvider: "exotel",
            websocket: opts.websocket,
            markEventMetaData: opts.markEventMetaData,
            logDirName: opts.logDirName,
        });
        (this as unknown as Record<string, unknown>)["isChunkingSupported"] = true;
    }

    // ------------------------------------------------------------------
    // Interruption
    // ------------------------------------------------------------------

    override async handleInterruption(): Promise<void> {
        if (this.isClosed()) return;
        try {
            logger.info("interrupting because user spoke in between");
            const messageClear = {
                event: "clear",
                stream_sid: this.streamSid,
            };
            await (this as unknown as { websocket: { send_text: (s: string) => Promise<void> } })
                .websocket.send_text(JSON.stringify(messageClear));
            (this as unknown as { markEventMetaData: MarkEventMetaData })
                .markEventMetaData?.clearData();
        } catch (e) {
            logger.info(`WebSocket closed during interruption: ${e}`);
            this.close();
        }
    }

    // ------------------------------------------------------------------
    // Media message
    // ------------------------------------------------------------------

    override async formMediaMessage(
        audioData: Buffer,
        audioFormat: string
    ): Promise<Record<string, unknown>> {
        let data = audioData;

        // Exotel expects PCM (16-bit linear) — convert from μ-law if needed
        if (audioFormat === "mulaw") {
            logger.info("Converting mulaw to PCM for Exotel");
            data = ulaw2lin(audioData);
        }

        const base64Audio = data.toString("base64");
        return {
            event: "media",
            stream_sid: this.streamSid,
            media: { payload: base64Audio },
        };
    }

    // ------------------------------------------------------------------
    // Mark message
    // ------------------------------------------------------------------

    override async formMarkMessage(markId: string): Promise<Record<string, unknown>> {
        return {
            event: "mark",
            stream_sid: this.streamSid,
            mark: { name: markId },
        };
    }

    override requiresCustomVoicemailDetection(): boolean {
        return false;
    }
}