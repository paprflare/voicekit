import { configureLogger } from "../../helper/logger";
import { TelephonyInputHandler } from "../telephony";
import type WebSocket from "ws";
import type { MarkEventMetaData } from "../../helper/markEvent.metadata";
import type { ObservableVariable } from "../../helper/observable.variable";

const logger = configureLogger("exotelInputHandler");

interface ObservableVariables {
    [key: string]: ObservableVariable<unknown> | undefined;
}

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

export class ExotelInputHandler extends TelephonyInputHandler {
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
        this.ioProvider = "exotel";
    }

    override async callStart(packet: Record<string, unknown>): Promise<void> {
        const start = packet["start"] as Record<string, string>;
        this.callSid = start["call_sid"] ?? null;
        this.streamSid = start["stream_sid"] ?? null;
    }

    override getMarkEventMetaDataObj(packet: Record<string, unknown>): Record<string, unknown> {
        const mark = packet["mark"] as Record<string, string>;
        const markId = mark["name"];
        return (this.markEventMetaData?.fetchData(markId as string) ?? {}) as Record<string, unknown>;
    }
}