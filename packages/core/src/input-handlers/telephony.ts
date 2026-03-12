import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket } from "../helper/utils";
import { DefaultInputHandler } from "./default";
import type { MarkEventMetaData } from "../helper/markEvent.metadata";
import type { ObservableVariable } from "../helper/observable.variable";

const logger = configureLogger("telephonyInputHandler");

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
    final_chunk_played_observable?: ObservableVariable<boolean>;
    agent_hangup_observable?: ObservableVariable<boolean>;
    init_event_observable?: ObservableVariable<boolean> | undefined;
}

export class TelephonyInputHandler extends DefaultInputHandler {
    protected streamSid: string | null = null;
    protected callSid: string | null = null;
    private buffer: Buffer[] = [];
    private messageCount = 0;
    private lastMediaReceived = 0;
    protected  override websocketListenTask: Promise<void> | null = null;

    constructor(opts: {
        queues: Queues;
        websocket?: WebSocket | null;
        inputTypes?: Record<string, unknown>;
        markEventMetaData?: MarkEventMetaData | null;
        turnBasedConversation?: boolean;
        isWelcomeMessagePlayed?: boolean;
        observableVariables?: ObservableVariables;
    }) {
        super({
            queues: opts.queues as never,
            websocket: opts.websocket,
            inputTypes: opts.inputTypes,
            markEventMetaData: opts.markEventMetaData,
            turnBasedConversation: opts.turnBasedConversation,
            isWelcomeMessagePlayed: opts.isWelcomeMessagePlayed,
            observableVariables: opts.observableVariables,
        });
        this.ioProvider = "";
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    override getStreamSid(): string {
        return this.streamSid ?? "";
    }

    getCallSid(): string {
        return this.callSid ?? "";
    }

    // ------------------------------------------------------------------
    // Lifecycle hooks (overridden by provider subclasses)
    // ------------------------------------------------------------------

    async callStart(_packet: Record<string, unknown>): Promise<void> { }

    async disconnectStream(): Promise<void> { }

    private async safeDisconnectStream(): Promise<void> {
        try {
            await this.disconnectStream();
        } catch (e) {
            logger.error(`Error in disconnect_stream: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Stop
    // ------------------------------------------------------------------

    override  async stopHandler(): Promise<void> {
        logger.info("stopping handler");
        this.running = false;

        // Fire-and-forget disconnect — don't block
        this.safeDisconnectStream().catch(() => { });

        logger.info("sleeping for 2 seconds so that whatever needs to pass is passed");
        await new Promise((resolve) => setTimeout(resolve, 2000));

        try {
            this.websocket?.close();
            logger.info("WebSocket connection closed");
        } catch (e) {
            logger.info(`Error closing WebSocket: ${e}`);
        }
    }

    // ------------------------------------------------------------------
    // Audio ingestion
    // ------------------------------------------------------------------

    async ingestAudio(
        audioData: Buffer,
        metaInfo: Record<string, unknown>
    ): Promise<void> {
        const wsDataPacket = createWsDataPacket({ data: audioData, metaInfo });
        (this.queues as unknown as Queues)["transcriber"].put_nowait(wsDataPacket);
    }

    // ------------------------------------------------------------------
    // DTMF
    // ------------------------------------------------------------------

    async handleDtmfDigit(digit: string): Promise<boolean> {
        if (!this.isDtmfActive) return false;
        const terminationKey = "#";
        if (digit === terminationKey) {
            logger.info("DTMF termination key pressed");
            return true;
        }
        this.dtmfDigits += digit;
        return false;
    }

    // ------------------------------------------------------------------
    // Listen loop
    // ------------------------------------------------------------------

    protected override listen(): Promise<void> {
        return new Promise<void>((resolve) => {
            const ws = this.websocket!;
            const buffer: Buffer[] = [];

            const sendEos = () => {
                const wsDataPacket = createWsDataPacket({
                    data: null,
                    metaInfo: { io: "default", eos: true },
                });
                (this.queues as unknown as Queues)["transcriber"].put_nowait(wsDataPacket);
            };

            const onMessage = async (raw: WebSocket.RawData) => {
                try {
                    const packet = JSON.parse(raw.toString()) as Record<string, unknown>;
                    const event = packet["event"] as string;

                    // ---- start ----
                    if (event === "start") {
                        await this.callStart(packet);

                        // ---- media ----
                    } else if (event === "media") {
                        const mediaData = packet["media"] as Record<string, unknown>;
                        const mediaAudio = Buffer.from(mediaData["payload"] as string, "base64");
                        const mediaTs = parseInt(String(mediaData["timestamp"] ?? "0"), 10);
                        const hasChunk = "chunk" in mediaData;
                        const isInbound =
                            "track" in mediaData && mediaData["track"] === "inbound";

                        if (hasChunk || isInbound) {
                            const metaInfo: Record<string, unknown> = {
                                io: this.ioProvider,
                                call_sid: this.callSid,
                                stream_sid: this.streamSid,
                                sequence: (this.inputTypes as Record<string, unknown>)["audio"],
                            };

                            this.lastMediaReceived = mediaTs;
                            buffer.push(mediaAudio);
                            this.messageCount += 1;

                            // Send 100ms of audio (10 × 10ms Twilio chunks) to transcriber
                            if (this.messageCount === 10) {
                                const mergedAudio = Buffer.concat(buffer);
                                buffer.length = 0;
                                await this.ingestAudio(mergedAudio, metaInfo);
                                this.messageCount = 0;
                            }
                        } else {
                            logger.info("Getting media elements but not inbound media");
                        }

                        // ---- mark / playedStream ----
                    } else if (event === "mark" || event === "playedStream") {
                        this.processMarkMessage(packet);

                        // ---- dtmf ----
                    } else if (event === "dtmf") {
                        const dtmfObj = (packet["dtmf"] ?? {}) as Record<string, unknown>;
                        const digit = (dtmfObj["digit"] as string) ?? "";
                        logger.info(`DTMF key pressed: '${digit}' | Accumulated: '${this.dtmfDigits}'`);
                        if (!digit) return;

                        const isComplete = await this.handleDtmfDigit(digit);
                        if (isComplete && this.dtmfDigits) {
                            if (this.isDtmfActive) {
                                logger.info(`DTMF complete - Sending: '${this.dtmfDigits}'`);
                                (this.queues as unknown as Queues)["dtmf"]?.put_nowait(this.dtmfDigits);
                            }
                            this.dtmfDigits = "";
                        }

                        // ---- stop ----
                    } else if (event === "stop") {
                        logger.info("call stopping");
                        sendEos();
                        cleanup();
                        resolve();
                    }
                } catch (e) {
                    console.error(e);
                    logger.info(`Exception in ${this.ioProvider} receiver reading events: ${e}`);
                    sendEos();
                    cleanup();
                    resolve();
                }
            };

            const onClose = (code: number, reason: Buffer) => {
                const normalCodes = new Set([1000, 1001, 1006]);
                if (!normalCodes.has(code)) {
                    logger.error(
                        `WebSocket disconnected unexpectedly: code=${code}, reason=${reason.toString()}`
                    );
                }
                sendEos();
                cleanup();
                resolve();
            };

            const onError = (err: Error) => {
                console.error(err);
                logger.info(`WebSocket error in ${this.ioProvider}: ${err}`);
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

    // ------------------------------------------------------------------
    // Handle
    // ------------------------------------------------------------------

    override async handle(): Promise<void> {
        if (!this.websocketListenTask) {
            this.websocketListenTask = this.listen();
        }
    }
}