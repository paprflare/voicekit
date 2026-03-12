import { configureLogger } from "../helper/logger";
import { Readable } from "stream";
import type { TurnLatency } from "../helper/utils";

const logger = configureLogger("baseSynthesizer");

// Simple async queue backed by a promise chain
export class AsyncQueue<T> {
    private queue: T[] = [];
    private waiters: ((value: T) => void)[] = [];

    put_nowait(item: T): void {
        const waiter = this.waiters.shift();
        if (waiter) {
            waiter(item);
        } else {
            this.queue.push(item);
        }
    }

    async put(item: T): Promise<void> {
        this.put_nowait(item);
    }

    async get(): Promise<T> {
        if (this.queue.length) return this.queue.shift()!;
        return new Promise<T>((resolve) => this.waiters.push(resolve));
    }

    get size(): number {
        return this.queue.length;
    }

    clear(): void {
        this.queue = [];
        this.waiters = [];
    }
}


export abstract class BaseSynthesizer {
    protected stream: boolean;
    protected bufferSize: number;
    protected internalQueue: AsyncQueue<unknown>;
    protected taskManagerInstance: Record<string, unknown> | null;
    protected connectionTime: number | null = null;
    protected turnLatencies: TurnLatency[] = [];

    constructor(opts: {
        taskManagerInstance?: Record<string, unknown> | null;
        stream?: boolean;
        bufferSize?: number;
    } = {}) {
        this.stream = opts.stream ?? true;
        this.bufferSize = opts.bufferSize ?? 40;
        this.internalQueue = new AsyncQueue();
        this.taskManagerInstance = opts.taskManagerInstance ?? null;
    }

    // ------------------------------------------------------------------
    // Queue
    // ------------------------------------------------------------------

    clearInternalQueue(): void {
        logger.info("Clearing out internal queue");
        this.internalQueue = new AsyncQueue();
    }

    shouldSynthesizeResponse(sequenceId: number): boolean {
        return (
            this.taskManagerInstance?.["is_sequence_id_in_current_ids"] as
            | ((id: number) => boolean)
            | undefined
        )?.(sequenceId) ?? false;
    }

    // ------------------------------------------------------------------
    // Abstract / overrideable stubs
    // ------------------------------------------------------------------

    async flushSynthesizerStream(): Promise<void> { }

    generate(): void { }

    push(_text: string): void { }

    synthesize(_text: string): void { }

    getSynthesizedCharacters(): number {
        return 0;
    }

    async monitorConnection(): Promise<void> { }

    async cleanup(): Promise<void> { }

    async handleInterruption(): Promise<void> { }

    // ------------------------------------------------------------------
    // Text helpers
    // ------------------------------------------------------------------

    *textChunker(text: string): Generator<string> {
        const splitters = new Set([".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " "]);
        let buffer = "";

        for (const char of text) {
            buffer += char;
            if (splitters.has(char)) {
                if (buffer !== " ") yield buffer.trim() + " ";
                buffer = "";
            }
        }

        if (buffer) yield buffer.trim() + " ";
    }

    normalizeText(s: string): string {
        return s.trim().replace(/\s+/g, " ");
    }

    // ------------------------------------------------------------------
    // Audio resampling (pydub → fluent-ffmpeg)
    // ------------------------------------------------------------------

    async resample(audioBytes: Buffer): Promise<Buffer> {
        const ffmpeg = (await import("fluent-ffmpeg")).default;

        return new Promise<Buffer>((resolve, reject) => {
            const chunks: Buffer[] = [];
            const input = new Readable({ read() { } });

            ffmpeg(input)
                .audioFrequency(8000)
                .audioChannels(1)
                .format("wav")
                .on("error", reject)
                .pipe()
                .on("data", (chunk: Buffer) => chunks.push(chunk))
                .on("end", () => resolve(Buffer.concat(chunks)))
                .on("error", reject);

            input.push(audioBytes);
            input.push(null);
        });
    }

    // ------------------------------------------------------------------
    // Provider info
    // ------------------------------------------------------------------

    getEngine(): string {
        return "default";
    }

    supportsWebsocket(): boolean {
        return true;
    }

    getSleepTime(): number {
        return 0.2;
    }
}