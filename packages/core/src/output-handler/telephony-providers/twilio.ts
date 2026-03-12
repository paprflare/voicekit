import { configureLogger } from "../../helper/logger";
import { TelephonyOutputHandler } from "../telephony";
import type { MarkEventMetaData } from "../../helper/markEvent.metadata";
import { lin2ulaw } from "../../helper/utils";

const logger = configureLogger("twilioOutputHandler");

export class TwilioOutputHandler extends TelephonyOutputHandler {
    constructor(opts: {
        websocket?: unknown;
        markEventMetaData?: MarkEventMetaData | null;
        logDirName?: string | null;
    } = {}) {
        super({
            ioProvider: "twilio",
            websocket: opts.websocket,
            markEventMetaData: opts.markEventMetaData,
            logDirName: opts.logDirName,
        });
        (this as unknown as Record<string, unknown>)["isChunkingSupported"] = true;
    }

    override async handleInterruption(): Promise<void> {
        if (this.isClosed()) return;
        try {
            logger.info("interrupting because user spoke in between");
            await (this as unknown as { websocket: { send_text: (s: string) => Promise<void> } })
                .websocket.send_text(
                    JSON.stringify({ event: "clear", streamSid: this.streamSid })
                );
            (this as unknown as { markEventMetaData: MarkEventMetaData })
                .markEventMetaData?.clearData();
        } catch (e) {
            logger.info(`WebSocket closed during interruption: ${e}`);
            this.close();
        }
    }

    override async formMediaMessage(
        audioData: Buffer,
        audioFormat = "wav"
    ): Promise<Record<string, unknown>> {
        let data = audioData;
        if (audioFormat !== "mulaw") {
            logger.info("Converting to mulaw");
            data = lin2ulaw(audioData);
        }
        return {
            event: "media",
            streamSid: this.streamSid,
            media: { payload: data.toString("base64") },
        };
    }

    override async formMarkMessage(markId: string): Promise<Record<string, unknown>> {
        return {
            event: "mark",
            streamSid: this.streamSid,
            mark: { name: markId },
        };
    }

    override requiresCustomVoicemailDetection(): boolean {
        return false;
    }
}