import { configureLogger } from "../helper/logger";
import { BaseAgent } from "./base";
import type { BaseLLM } from "../llms/llm";

const logger = configureLogger("extractionContextualAgent");

export class ExtractionContextualAgent extends BaseAgent {
    private llm: BaseLLM;
    private currentMessages = 0;
    private isInferenceOn = false;
    private hasIntroBeenSent = false;

    constructor(llm: BaseLLM) {
        super();
        this.llm = llm;
    }

    async generate(history: Record<string, unknown>[]): Promise<unknown> {
        return this.llm.generate(history, false, false);
    }
}