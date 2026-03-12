import { configureLogger } from "../../helper/logger";
import { TelephonyOutputHandler } from "../telephony";
import type { MarkEventMetaData } from "../../helper/markEvent.metadata";

const logger = configureLogger("plivoOutputHandler");

export class PlivoOutputHandler extends TelephonyOutputHandler {
    constructor(opts: {
        websocket?: unknown;
        markEventMetaData?: MarkEventMetaData | null;
        logDirName?: string | null;
    } = {}) {
        super({
            ioProvider: "plivo",
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
                    JSON.stringify({ event: "clearAudio", streamId: this.streamSid })
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
        audioFormat = "audio/x-mulaw"
    ): Promise<Record<string, unknown>> {
        return {
            event: "playAudio",
            media: {
                payload: audioData.toString("base64"),
                sampleRate: "8000",
                contentType: audioFormat === "wav" ? "wav" : "audio/x-mulaw",
            },
        };
    }

    override async formMarkMessage(markId: string): Promise<Record<string, unknown>> {
        return {
            event: "checkpoint",
            streamId: this.streamSid,
            name: markId,
        };
    }
}