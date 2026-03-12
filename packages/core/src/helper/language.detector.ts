import { configureLogger } from "../helper/logger";
import { OpenAiLLM } from "../llms/openai.llm";
import { LANGUAGE_DETECTION_PROMPT } from "../prompts";
import { convertToRequestLog } from "../helper/utils";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("languageDetector");

interface LanguageDetectorConfig {
    language_detection_turns?: number;
}

interface DetectionResult {
    dominant_language: string;
    confidence: number;
    reasoning: string;
}

export class LanguageDetector {
    private turnsThreshold: number;
    private runId: string;

    private transcripts: string[] = [];
    private result: DetectionResult | null = null;
    private complete = false;
    private inProgress = false;
    private task: Promise<void> | null = null;
    private llm: OpenAiLLM | null = null;
    private latencyMs: number | null = null;

    constructor(config: LanguageDetectorConfig, runId = "") {
        this.turnsThreshold = config.language_detection_turns ?? 0;
        this.runId = runId;

        if (this.turnsThreshold > 0) {
            this.llm = new OpenAiLLM(
                100,
                40,
                process.env.LANGUAGE_DETECTION_LLM ?? "gpt-4.1-mini"
            );
        }
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    get isEnabled(): boolean {
        return this.turnsThreshold > 0;
    }

    get dominantLanguage(): string | null {
        return this.complete && this.result
            ? this.result.dominant_language
            : null;
    }

    get latencyData(): Record<string, unknown> | null {
        if (!this.complete || this.latencyMs === null) return null;
        return {
            type: "language_detection",
            latency_ms: this.latencyMs,
            model: this.llm?.getModel ?? null,
            provider: "openai",
        };
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /** Collect transcript and trigger detection after N turns. */
    async collectTranscript(transcript: string): Promise<void> {
        if (this.complete || !this.turnsThreshold || this.inProgress) return;

        this.transcripts.push(transcript);
        logger.info(
            `Language detection: collected ${this.transcripts.length}/${this.turnsThreshold} transcripts`
        );

        if (this.transcripts.length >= this.turnsThreshold) {
            this.inProgress = true;
            this.task = this.runDetection();
        }
    }

    // ------------------------------------------------------------------
    // Private
    // ------------------------------------------------------------------

    private async runDetection(): Promise<void> {
        try {
            const formatted = this.transcripts.map((t) => `- ${t}`).join("\n");
            const prompt = LANGUAGE_DETECTION_PROMPT(formatted);

            const startTime = Date.now();
            const response = await this.llm!.generate(
                [{ role: "system", content: prompt }],
                false,
                false
            );
            this.latencyMs = Date.now() - startTime;

            this.result = JSON.parse(response as string) as DetectionResult;
            this.complete = true;
            logger.info(`Language detection complete: ${JSON.stringify(this.result)}`);
            this.logDetection(this.result);
        } catch (e) {
            logger.error(`Language detection error: ${e}`);
            this.result = null;
            this.complete = true;
        } finally {
            this.inProgress = false;
            this.task = null;
        }
    }

    private logDetection(result: DetectionResult): void {
        const metaInfo = { request_id: uuidv4() };
        const model = this.llm?.getModel ?? "unknown";

        convertToRequestLog({
            message: { transcripts: this.transcripts },
            metaInfo: metaInfo as never,
            component: "llm_language_detection",
            direction: "request",
            model,
            runId: this.runId,
        });

        convertToRequestLog({
            message: result,
            metaInfo: metaInfo as never,
            component: "llm_language_detection",
            direction: "response",
            model,
            runId: this.runId,
        });
    }
}