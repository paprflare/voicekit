import { configureLogger } from "../../helper/logger";
import { TelephonyInputHandler } from "../telephony";
import type WebSocket from "ws";
import type { MarkEventMetaData } from "../../helper/markEvent.metadata";
import type { ObservableVariable } from "../../helper/observable.variable";
import plivo from "plivo";

const logger = configureLogger("plivoInputHandler");

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
    [key: string]: ObservableVariable<unknown> | undefined;
}

export class PlivoInputHandler extends TelephonyInputHandler {
    private client: plivo.Client;

    constructor(opts: {
        queues: Queues;
        websocket?: WebSocket | null;
        inputTypes?: Record<string, unknown>;
        markEventMetaData?: MarkEventMetaData | null;
        turnBasedConversation?: boolean;
        isWelcomeMessagePlayed?: boolean;
        observableVariables?: ObservableVariables;
    }) {
        super(opts as never);
        this.ioProvider = "plivo";
        this.client = new plivo.Client(
            process.env.PLIVO_AUTH_ID!,
            process.env.PLIVO_AUTH_TOKEN!
        );
    }

    override async callStart(packet: Record<string, unknown>): Promise<void> {
        const start = packet["start"] as Record<string, string>;
        this.callSid = start["callId"] ?? null;
        this.streamSid = start["streamId"] ?? null;
    }

    override async disconnectStream(): Promise<void> {
        try {
            await this.client.calls.hangup(this.callSid!);
        } catch (e) {
            logger.info(`Error deleting plivo stream: ${e}`);
        }
    }

    override getMarkEventMetaDataObj(packet: Record<string, unknown>): Record<string, unknown> {
        const markId = packet["name"] as string;
        return (this.markEventMetaData?.fetchData(markId) ?? {}) as Record<string, unknown>;
    }
}