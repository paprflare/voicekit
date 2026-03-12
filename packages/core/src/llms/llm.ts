

// llm.ts
export abstract class BaseLLM {
    protected bufferSize: number;
    protected maxTokens: number;

    constructor(maxTokens = 100, bufferSize = 40) {
        this.maxTokens = maxTokens;
        this.bufferSize = bufferSize;
    }

    async respondBackWithFiller(messages: Record<string, unknown>[]): Promise<void> { }

    abstract generate(
        messages: Record<string, unknown>[],
        stream?: boolean,
        retMetadata?: boolean
    ): Promise<unknown>;

    invalidateResponseChain(): void { }
}