/**
 * Asterisk WebSocket (chan_websocket) output handler for sip-trunk provider.
 * Send all media between START_MEDIA_BUFFERING and STOP_MEDIA_BUFFERING; Asterisk re-frames and re-times.
 * On MEDIA_XOFF queue locally; on MEDIA_XON drain to Asterisk. FLUSH_MEDIA on interrupt.
 * Ref: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
 */

import { configureLogger } from "../../helper/logger";
import { TelephonyOutputHandler } from "../telephony";
import type { MarkEventMetaData } from "../../helper/markEvent.metadata";
import { lin2ulaw } from "../../helper/utils";

const logger = configureLogger("sipTrunkOutputHandler");

const ASTERISK_ULAW_OPTIMAL_FRAME_SIZE = 160;
const MAX_WS_MESSAGE_BYTES = 65500;
const PREFERRED_WS_SEND_CHUNK_BYTES = 8192;
const PLAYBACK_DONE_BUFFER_S = 0.5;

interface MetaInfo {
    stream_sid?: string;
    end_of_llm_stream?: boolean;
    end_of_synthesizer_stream?: boolean;
    is_final_chunk_of_entire_response?: boolean;
    sequence_id?: number;
    format?: string;
    message_category?: string;
    mark_id?: string;
    is_first_chunk?: boolean;
    text_synthesized?: string;
    cached?: boolean;
}

interface Packet {
    data: Buffer | string;
    meta_info: MetaInfo;
}

interface AsteriskWebSocket {
    send_text(data: string): Promise<void>;
    send_bytes(data: Buffer): Promise<void>;
}

export class SipTrunkOutputHandler extends TelephonyOutputHandler {
    private asteriskMediaStart: Record<string, unknown>;
    private agentConfig: Record<string, unknown>;
    private optimalFrameSize: number;
    private inputHandler: Record<string, unknown> | null;
    queueFull = false;

    private bufferingActive = false;
    private responseAudioDuration = 0.0;
    private playbackDoneTask: ReturnType<typeof setTimeout> | null = null;
    private playbackDoneAbort: AbortController | null = null;
    private localAudioQueue: Buffer[] = [];
    private pendingStopAfterDrain = false;
    private pendingStopDuration = 0.0;
    private pendingStopCategory = "agent_response";
    private outputFormat: string;

    constructor(opts: {
        ioProvider?: string;
        websocket?: unknown;
        markEventMetaData?: MarkEventMetaData | null;
        logDirName?: string | null;
        asteriskMediaStart?: Record<string, unknown>;
        agentConfig?: Record<string, unknown>;
        inputHandler?: Record<string, unknown> | null;
    } = {}) {
        super({
            ioProvider: opts.ioProvider ?? "sip-trunk",
            websocket: opts.websocket,
            markEventMetaData: opts.markEventMetaData,
            logDirName: opts.logDirName,
        });

        this.asteriskMediaStart = opts.asteriskMediaStart ?? {};
        this.agentConfig = opts.agentConfig ?? {};
        this.optimalFrameSize = ASTERISK_ULAW_OPTIMAL_FRAME_SIZE;
        this.inputHandler = opts.inputHandler ?? null;

        if (this.inputHandler) {
            this.inputHandler["output_handler_ref"] = this;
        }

        const outputConfig = this.getOutputConfig();
        const fmt = (
            (outputConfig["audio_format"] as string) ||
            (outputConfig["format"] as string) ||
            "ulaw"
        ).toLowerCase();
        this.outputFormat = ["ulaw", "mulaw"].includes(fmt) ? fmt : "ulaw";

        const opt = this.asteriskMediaStart["optimal_frame_size"];
        if (opt != null) {
            const parsed = parseInt(String(opt), 10);
            if (!isNaN(parsed)) this.optimalFrameSize = parsed;
        }
    }

    // ------------------------------------------------------------------
    // Config helpers
    // ------------------------------------------------------------------

    private getOutputConfig(): Record<string, unknown> {
        try {
            const tasks = (this.agentConfig["tasks"] as Record<string, unknown>[]) ?? [];
            if (tasks.length && typeof tasks[0] === "object") {
                return (
                    ((tasks[0]!["tools_config"] as Record<string, unknown>)?.["output"] as Record<string, unknown>) ?? {}
                );
            }
        } catch { /* ignore */ }
        return {};
    }

    // ------------------------------------------------------------------
    // Interruption
    // ------------------------------------------------------------------

    override async handleInterruption(): Promise<void> {
        logger.info("sip-trunk: handling interruption (FLUSH_MEDIA)");
        try {
            this.cancelPlaybackDoneTask();
            this.bufferingActive = false;
            this.responseAudioDuration = 0.0;
            this.localAudioQueue = [];
            this.pendingStopAfterDrain = false;
            await this.sendControl("FLUSH_MEDIA");
            (this as unknown as { markEventMetaData: MarkEventMetaData })
                .markEventMetaData?.clearData();
            if (this.inputHandler) {
                (this.inputHandler["update_is_audio_being_played"] as (v: boolean) => void)?.(false);
            }
        } catch (e) {
            logger.error(`sip-trunk handle_interruption: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Drain local queue (called when MEDIA_XON received)
    // ------------------------------------------------------------------

    async drainLocalQueue(): Promise<void> {
        while (this.localAudioQueue.length && !this.queueFull) {
            const chunk = this.localAudioQueue.shift()!;
            if (!chunk?.length) continue;
            await this.sendBinary(chunk);
        }

        if (!this.localAudioQueue.length && this.pendingStopAfterDrain) {
            this.pendingStopAfterDrain = false;
            if (this.bufferingActive) {
                await this.sendControl("STOP_MEDIA_BUFFERING");
                this.bufferingActive = false;
            }
            await this.sendControl("REPORT_QUEUE_DRAINED");
            const duration = this.pendingStopDuration;
            const category = this.pendingStopCategory;
            this.cancelPlaybackDoneTask();
            this.schedulePlaybackDoneFallback(duration, category);
        }
    }

    // ------------------------------------------------------------------
    // Low-level send helpers
    // ------------------------------------------------------------------

    private async sendControl(command: string, params?: Record<string, unknown>): Promise<void> {
        try {
            const msg = params
                ? `${command} ${Object.values(params).join(" ")}`
                : command;
            await (this as unknown as { websocket: AsteriskWebSocket }).websocket.send_text(msg);
            logger.debug(`sip-trunk sent: ${command}`);
        } catch (e) {
            logger.error(`sip-trunk send_control ${command}: ${e}`);
        }
    }

    private async sendBinary(data: Buffer): Promise<void> {
        const ws = (this as unknown as { websocket: AsteriskWebSocket }).websocket;
        const n = data.length;
        if (n <= PREFERRED_WS_SEND_CHUNK_BYTES) {
            await ws.send_bytes(data);
            return;
        }
        let offset = 0;
        while (offset < n) {
            const chunkSize = Math.min(PREFERRED_WS_SEND_CHUNK_BYTES, n - offset, MAX_WS_MESSAGE_BYTES);
            const chunk = data.subarray(offset, offset + chunkSize);
            if (chunk.length) await ws.send_bytes(chunk);
            offset += chunkSize;
        }
    }

    async flushMedia(): Promise<void> {
        await this.sendControl("FLUSH_MEDIA");
    }

    // ------------------------------------------------------------------
    // Stubs (Asterisk uses binary, not JSON messages)
    // ------------------------------------------------------------------

    override async formMediaMessage(_audioData: Buffer, _audioFormat = "wav"): Promise<Record<string, unknown>> {
        return {};
    }

    override async formMarkMessage(_markId: string): Promise<Record<string, unknown>> {
        return {};
    }

    override async setStreamSid(streamId: string): Promise<void> {
        this.streamSid = streamId;
    }

    // ------------------------------------------------------------------
    // Duration helper
    // ------------------------------------------------------------------

    private durationUlaw(numBytes: number): number {
        return numBytes / 8000.0;
    }

    // ------------------------------------------------------------------
    // Main handle
    // ------------------------------------------------------------------

    override async handle(wsDataPacket: Packet): Promise<void> {
        try {
            let audioChunk = wsDataPacket.data;
            const metaInfo: MetaInfo = wsDataPacket.meta_info ?? {};

            if (!this.streamSid) {
                this.streamSid = metaInfo.stream_sid ?? null;
            }

            const isFinal = Boolean(
                (metaInfo.end_of_llm_stream && metaInfo.end_of_synthesizer_stream) ||
                metaInfo.is_final_chunk_of_entire_response ||
                (metaInfo.sequence_id === -1 && metaInfo.end_of_llm_stream)
            );

            const hasAudio =
                audioChunk &&
                audioChunk.length > 1 &&
                !(audioChunk.length === 2 && audioChunk[0] === 0x00 && audioChunk[1] === 0x00);

            if (!hasAudio && !isFinal) return;

            let audioFormat = (metaInfo.format ?? "ulaw").toLowerCase();
            let audioDuration = 0.0;

            if (hasAudio && audioChunk) {
                // Pad single-byte chunk
                if (audioChunk.length === 1) {
                    audioChunk = Buffer.concat([Buffer.from(audioChunk), Buffer.from([0x00])]);
                }

                // PCM/WAV → μ-law conversion
                const isWavHeader = audioChunk.length > 44 && (audioChunk as Buffer).subarray(0, 4).toString() === "RIFF";
                if (audioFormat === "pcm" || audioFormat === "wav" || isWavHeader) {
                    if (isWavHeader) audioChunk = (audioChunk as Buffer).subarray(44);
                    audioChunk = lin2ulaw(audioChunk as Buffer);
                    audioFormat = "ulaw";
                }

                if (
                    metaInfo.message_category === "agent_welcome_message" &&
                    !this.getWelcomeMessageSentTs()
                ) {
                    (this as unknown as Record<string, unknown>)["welcomeMessageSentTs"] = Date.now();
                }

                // START_MEDIA_BUFFERING before first chunk of a new response
                if (!this.bufferingActive && audioChunk.length > this.optimalFrameSize) {
                    await this.sendControl("START_MEDIA_BUFFERING");
                    this.bufferingActive = true;
                    this.responseAudioDuration = 0.0;
                }

                if (this.queueFull) {
                    this.localAudioQueue.push(audioChunk as Buffer);
                } else {
                    await this.sendBinary(audioChunk as Buffer);
                }

                audioDuration = this.durationUlaw(audioChunk.length);
                this.responseAudioDuration += audioDuration;
            }

            // Mark tracking
            const markEventMetaData = (this as unknown as { markEventMetaData: MarkEventMetaData }).markEventMetaData;
            if (markEventMetaData) {
                const messageCategory = metaInfo.message_category ?? "agent_response";
                const markId = (metaInfo.mark_id && metaInfo.mark_id !== "") ? metaInfo.mark_id : crypto.randomUUID();

                markEventMetaData.updateData(markId, {
                    text_synthesized: metaInfo.sequence_id !== -1 ? (metaInfo.text_synthesized ?? "") : "",
                    type: messageCategory,
                    is_first_chunk: metaInfo.is_first_chunk ?? false,
                    is_final_chunk: isFinal,
                    sequence_id: metaInfo.sequence_id ?? 0,
                    duration: audioDuration,
                    sent_ts: Date.now() / 1000,
                });

                if (isFinal) {
                    const totalDuration = this.responseAudioDuration;

                    if (this.localAudioQueue.length) {
                        this.pendingStopAfterDrain = true;
                        this.pendingStopDuration = totalDuration;
                        this.pendingStopCategory = messageCategory;
                        logger.debug("sip-trunk: final chunk queued (XOFF); will send STOP after drain");
                    } else {
                        if (this.bufferingActive) {
                            await this.sendControl("STOP_MEDIA_BUFFERING");
                            this.bufferingActive = false;
                        }
                        await this.sendControl("REPORT_QUEUE_DRAINED");
                        if (totalDuration > 0 || !this.playbackDoneTask) {
                            this.cancelPlaybackDoneTask();
                            this.schedulePlaybackDoneFallback(totalDuration, messageCategory);
                        }
                        logger.debug(`sip-trunk: response done, fallback in ${(totalDuration + PLAYBACK_DONE_BUFFER_S).toFixed(1)}s`);
                    }
                }
            }
        } catch (e) {
            logger.error(`sip-trunk output error: ${e}`);
            console.error(e);
        }
    }

    // ------------------------------------------------------------------
    // Playback done fallback
    // ------------------------------------------------------------------

    private cancelPlaybackDoneTask(): void {
        if (this.playbackDoneAbort) {
            this.playbackDoneAbort.abort();
            this.playbackDoneAbort = null;
        }
        if (this.playbackDoneTask) {
            clearTimeout(this.playbackDoneTask);
            this.playbackDoneTask = null;
        }
    }

    private schedulePlaybackDoneFallback(duration: number, messageCategory: string): void {
        const abort = new AbortController();
        this.playbackDoneAbort = abort;
        const delayMs = (duration + PLAYBACK_DONE_BUFFER_S) * 1000;
        this.playbackDoneTask = setTimeout(() => {
            if (!abort.signal.aborted) {
                this.playbackDoneFallback(duration, messageCategory);
            }
        }, delayMs);
    }

    private playbackDoneFallback(duration: number, messageCategory: string): void {
        this.playbackDoneTask = null;
        if (!this.inputHandler) return;

        const markData = this.inputHandler["mark_event_meta_data"] as MarkEventMetaData | undefined;
        if (!markData) return;

        const remaining = Object.keys(markData.fetchClearedMarkEventData?.() ?? markData["markEventMetaData"] ?? {});
        if (!remaining.length) return;

        logger.info(`sip-trunk: playback-done fallback (${duration.toFixed(2)}s), processing ${remaining.length} mark(s)`);
        (this.inputHandler["update_is_audio_being_played"] as (v: boolean) => void)?.(false);

        for (const mid of remaining) {
            const md = (markData["markEventMetaData"] as Record<string, Record<string, unknown>>)?.[mid] ?? {};
            (this.inputHandler["process_mark_message"] as (m: Record<string, unknown>) => void)?.({
                name: mid,
                type: md["type"] ?? messageCategory,
            });
        }
    }

    // ------------------------------------------------------------------
    // Hangup
    // ------------------------------------------------------------------

    async sendHangup(): Promise<void> {
        await this.sendControl("HANGUP");
    }

    override setHangupSent(): void {
        super.setHangupSent();
        try {
            this.sendHangup().catch((e) =>
                logger.error(`sip-trunk send_hangup: ${e}`)
            );
        } catch (e) {
            logger.error(`sip-trunk send_hangup: ${e}`);
        }
    }

    override requiresCustomVoicemailDetection(): boolean {
        return false;
    }
}