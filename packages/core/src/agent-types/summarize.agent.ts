import { configureLogger } from "../helper/logger";
import { BaseAgent } from "./base";
import type { BaseLLM } from "../llms/llm";

const logger = configureLogger("summarizationContextualAgent");

export class SummarizationContextualAgent extends BaseAgent {
    private llm: BaseLLM;
    private currentMessages = 0;
    private isInferenceOn = false;
    private hasIntroBeenSent = false;

    constructor(llm: BaseLLM) {
        super();
        this.llm = llm;
    }

    async generate(history: Record<string, unknown>[]): Promise<{ summary: string }> {
        let summary = "";
        try {
            summary = (await this.llm.generate(history as never, false, false)) as string;
        } catch (e) {
            console.error(e);
            logger.error(`error in generating summary: ${e}`);
        }
        return { summary };
    }
}