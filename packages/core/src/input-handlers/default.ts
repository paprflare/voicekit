import WebSocket from "ws";
import { configureLogger } from "../helper/logger";
import { createWsDataPacket } from "../helper/utils";
import type { MarkEventMetaData } from "../helper/markEvent.metadata";
import type { ObservableVariable } from "../helper/observable.variable";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("defaultInputHandler");

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
    [key: string]: AsyncQueue;
}

interface ConversationRecording {
    metadata: { started: number };
    input: { data: Buffer };
}

interface ObservableVariables {
    [key: string]: ObservableVariable<boolean> | undefined;
    final_chunk_played_observable?: ObservableVariable<boolean>;
    agent_hangup_observable?: ObservableVariable<boolean>;
    init_event_observable?: ObservableVariable<boolean> | undefined;
}

interface MarkEventMetaDataObj {
    type?: string;
    text_synthesized?: string;
    is_final_chunk?: boolean;
    sent_ts?: number;
    duration?: number;
}

// ============================================================
// DefaultInputHandler
// ============================================================

export class DefaultInputHandler {
    protected queues: Queues | null;
    protected websocket: WebSocket | null;
    protected inputTypes: Record<string, unknown>;
    protected websocketListenTask: Promise<void> | null = null;
    running = true;
    private turnBasedConversation: boolean;
    protected queue: AsyncQueue | null;
    private conversationRecording: ConversationRecording | null;
    isWelcomeMessagePlayed: boolean;
    responseHeardByUser = "";
    protected _isAudioBeingPlayedToUser = false;
    protected observableVariables: ObservableVariables;
    markEventMetaData: MarkEventMetaData | null;
    audioChunksReceived = 0;
    protected updateStartTs: number;
    protected ioProvider = "default";
    isDtmfActive = false;
    dtmfDigits = "";
    private plivoLatencySamples: number[] = [];
    private calculatedPlivoLatency = 0.25;
    private readonly maxLatencySamples = 10;

    // Resolved once the websocket 'close' or 'error' fires
    private disconnectResolve: (() => void) | null = null;
    private disconnectPromise: Promise<void>;

    constructor(opts: {
        queues?: Queues | null;
        websocket?: WebSocket | null;
        inputTypes?: Record<string, unknown>;
        markEventMetaData?: MarkEventMetaData | null;
        queue?: AsyncQueue | null;
        turnBasedConversation?: boolean;
        conversationRecording?: ConversationRecording | null;
        isWelcomeMessagePlayed?: boolean;
        observableVariables?: ObservableVariables;
    } = {}) {
        this.queues = opts.queues ?? null;
        this.websocket = opts.websocket ?? null;
        this.inputTypes = opts.inputTypes ?? {};
        this.markEventMetaData = opts.markEventMetaData ?? null;
        this.queue = opts.queue ?? null;
        this.turnBasedConversation = opts.turnBasedConversation ?? false;
        this.conversationRecording = opts.conversationRecording ?? null;
        this.isWelcomeMessagePlayed = opts.isWelcomeMessagePlayed ?? false;
        this.observableVariables = opts.observableVariables ?? {};
        this.updateStartTs = Date.now() / 1000;

        // Create a promise that resolves when the WS disconnects
        this.disconnectPromise = new Promise((resolve) => {
            this.disconnectResolve = resolve;
        });

        if (this.websocket) {
            this.websocket.once("close", () => this.disconnectResolve?.());
            this.websocket.once("error", () => this.disconnectResolve?.());
        }
    }

    // ------------------------------------------------------------------
    // Send helpers (matching the interface expected by output handlers)
    // ------------------------------------------------------------------

    async send_text(data: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.websocket?.send(data, (err) => (err ? reject(err) : resolve()));
        });
    }

    async send_json(data: unknown): Promise<void> {
        return this.send_text(JSON.stringify(data));
    }

    async send_bytes(data: Buffer): Promise<void> {
        return new Promise((resolve, reject) => {
            this.websocket?.send(data, { binary: true }, (err) =>
                err ? reject(err) : resolve()
            );
        });
    }

    // ------------------------------------------------------------------
    // Latency
    // ------------------------------------------------------------------

    getCalculatedPlivoLatency(): number {
        return this.calculatedPlivoLatency;
    }

    private calculateAndUpdateLatency(obj: MarkEventMetaDataObj): void {
        const sentTs = obj.sent_ts ?? 0;
        const duration = obj.duration ?? 0;
        if (sentTs <= 0) return;

        const latency = Date.now() / 1000 - sentTs - duration;
        if (latency < 0 || latency > 2.0) return;

        this.plivoLatencySamples.push(latency);
        if (this.plivoLatencySamples.length > this.maxLatencySamples) {
            this.plivoLatencySamples.shift();
        }
        this.calculatedPlivoLatency =
            this.plivoLatencySamples.reduce((a, b) => a + b, 0) /
            this.plivoLatencySamples.length;
    }

    // ------------------------------------------------------------------
    // Audio state
    // ------------------------------------------------------------------

    getAudioChunksReceived(): number {
        const count = this.audioChunksReceived;
        this.audioChunksReceived = 0;
        return count;
    }

    updateIsAudioBeingPlayed(value: boolean): void {
        logger.info(`Audio is being updated - ${value}`);
        if (value) {
            this.updateStartTs = Date.now() / 1000;
            logger.info(`updating ts as mark_message received: ${this.updateStartTs}`);
        }
        this._isAudioBeingPlayedToUser = value;
    }

    isAudioBeingPlayedToUser(): boolean {
        return this._isAudioBeingPlayedToUser;
    }

    // ------------------------------------------------------------------
    // Response tracking
    // ------------------------------------------------------------------

    getResponseHeardByUser(): string {
        const response = this.responseHeardByUser;
        this.responseHeardByUser = "";
        return response.trim();
    }

    resetResponseHeardByUser(): void {
        this.responseHeardByUser = "";
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    async stopHandler(): Promise<void> {
        this.running = false;
        try {
            if (!this.queue) {
                this.websocket?.close();
            }
        } catch (e) {
            logger.error(`Error closing WebSocket: ${e}`);
        }
    }

    getStreamSid(): string {
        return uuidv4();
    }

    getCurrentMarkStartedTime(): number {
        return this.updateStartTs;
    }

    welcomeMessagePlayed(): boolean {
        return this.isWelcomeMessagePlayed;
    }

    // ------------------------------------------------------------------
    // Mark events
    // ------------------------------------------------------------------

    getMarkEventMetaDataObj(packet: Record<string, unknown>): MarkEventMetaDataObj {
        const markId = packet["name"] as string;
        return (this.markEventMetaData?.fetchData(markId) ?? {}) as MarkEventMetaDataObj;
    }

    processMarkMessage(packet: Record<string, unknown>): void {
        const obj = this.getMarkEventMetaDataObj(packet);
        if (!obj || !Object.keys(obj).length) {
            logger.info(`No object retrieved from mark_event_meta_data for mark event - ${JSON.stringify(packet)}`);
            return;
        }

        const messageType = obj.type;
        const isContentAudio = !["ambient_noise", "backchanneling"].includes(messageType ?? "");

        if (messageType === "pre_mark_message") {
            this.updateIsAudioBeingPlayed(true);
            return;
        }

        this.audioChunksReceived += 1;
        this.calculateAndUpdateLatency(obj);

        if (isContentAudio) {
            this.responseHeardByUser += obj.text_synthesized ?? "";
        }

        if (obj.is_final_chunk) {
            if (messageType !== "is_user_online_message") {
                const obs = this.observableVariables["final_chunk_played_observable"];
                if (obs) obs.value = !(obs.value as boolean);
            }
            this.updateIsAudioBeingPlayed(false);

            if (messageType === "agent_welcome_message") {
                logger.info("Received mark event for agent_welcome_message");
                this.audioChunksReceived = 0;
                this.isWelcomeMessagePlayed = true;
            } else if (messageType === "agent_hangup") {
                logger.info("Agent hangup has been triggered");
                const obs = this.observableVariables["agent_hangup_observable"];
                if (obs) obs.value = true;
            }
        }
    }

    // ------------------------------------------------------------------
    // Message processing
    // ------------------------------------------------------------------

    private processAudio(audio: string): void {
        const data = Buffer.from(audio, "base64");
        const wsDataPacket = createWsDataPacket({
            data,
            metaInfo: {
                io: "default",
                type: "audio",
                sequence: this.inputTypes["audio"],
            },
        });

        if (this.conversationRecording) {
            if (this.conversationRecording.metadata.started === 0) {
                this.conversationRecording.metadata.started = Date.now() / 1000;
            }
            this.conversationRecording.input.data = Buffer.concat([
                this.conversationRecording.input.data,
                data,
            ]);
        }

        this.queues!["transcriber"].put_nowait(wsDataPacket);
    }

    private processText(text: string): void {
        logger.info(`Sequences ${JSON.stringify(this.inputTypes)}`);
        const wsDataPacket = createWsDataPacket({
            data: text,
            metaInfo: {
                io: "default",
                type: "text",
                sequence: this.inputTypes["audio"],
            },
        });

        if (this.turnBasedConversation) {
            (wsDataPacket["meta_info"] as Record<string, unknown>)["bypass_synth"] = true;
        }

        this.queues!["llm"].put_nowait(wsDataPacket);
    }

    async processMessage(message: Record<string, unknown>): Promise<Record<string, unknown> | void> {
        if (message["type"] === "audio") {
            this.processAudio(message["data"] as string);
        } else if (message["type"] === "text") {
            logger.info(`Received text: ${message["data"]}`);
            this.processText(message["data"] as string);
        } else if (message["type"] === "mark") {
            logger.info("Received mark event");
            this.processMarkMessage(message);
        } else if (message["type"] === "init") {
            logger.info("Received init event");
            const obs = this.observableVariables["init_event_observable"];
            if (obs) obs.value = message["meta_data"] as boolean ?? null;
        } else {
            return { message: "Other modalities not implemented yet" };
        }
    }

    // ------------------------------------------------------------------
    // Listen loop — event-driven via ws `message` event
    // ------------------------------------------------------------------

    protected async listen(): Promise<void> {
        if (this.queue !== null) {
            // Queue-based mode (no WebSocket)
            while (this.running) {
                try {
                    const request = (await this.queue.get()) as Record<string, unknown>;
                    await this.processMessage(request);
                } catch (e) {
                    logger.info(`Queue read error: ${e}`);
                    break;
                }
            }
            return;
        }

        // WebSocket mode — use event emitter pattern from `ws`
        return new Promise<void>((resolve) => {
            const ws = this.websocket!;

            const onMessage = (raw: WebSocket.RawData) => {
                if (!this.running) return;
                try {
                    const message = JSON.parse(raw.toString()) as Record<string, unknown>;
                    this.processMessage(message).catch((e) =>
                        logger.info(`Error processing message: ${e}`)
                    );
                } catch (e) {
                    logger.info(`Error parsing WebSocket message: ${e}`);
                }
            };

            const onClose = () => {
                this.running = false;
                sendEos();
                cleanup();
                resolve();
            };

            const onError = (err: Error) => {
                logger.info(`WebSocket error: ${err}`);
                this.running = false;
                sendEos();
                cleanup();
                resolve();
            };

            const cleanup = () => {
                ws.off("message", onMessage);
                ws.off("close", onClose);
                ws.off("error", onError);
            };

            const sendEos = () => {
                const wsDataPacket = createWsDataPacket({
                    data: null,
                    metaInfo: { io: "default", eos: true },
                });
                try {
                    this.queues!["transcriber"].put_nowait(wsDataPacket);
                } catch {
                    this.queues!["transcriber"].put(wsDataPacket).catch(() => { });
                }
            };

            ws.on("message", onMessage);
            ws.once("close", onClose);
            ws.once("error", onError);
        });
    }

    async handle(): Promise<void> {
        this.websocketListenTask = this.listen();
    }
}