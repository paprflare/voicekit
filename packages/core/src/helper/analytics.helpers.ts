import { configureLogger } from "./logger";
import { formatMessages } from "./utils";
import { CHECK_FOR_COMPLETION_PROMPT } from "../prompts";
import { HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA, type HighLevelAssistantAnalyticsData } from "../constants";
import { parseISO, differenceInMinutes } from "date-fns";
import { encoding_for_model, type TiktokenModel } from "tiktoken";

const logger = configureLogger("analyticsHelper");

// ============================================================
// Types
// ============================================================

interface Message {
    role: string;
    content: string | null;
}

interface RunDetails {
    extracted_data?: Record<string, string> | null;
    total_cost: number;
    conversation_time: number;
    cost_breakdown: {
        transcriber: number;
        synthesizer: number;
        llm: number;
    };
}

interface LlmTokenUsage {
    [model: string]: { input: number; output: number };
}

// ============================================================
// Token counting (replaces litellm.token_counter)
// ============================================================

function countTokens(model: string, text?: string, messages?: Message[]): number {
    try {
        const enc = encoding_for_model(model as TiktokenModel);
        if (text) {
            const count = enc.encode(text).length;
            enc.free();
            return count;
        }
        if (messages) {
            // 4 tokens overhead per message (role + separators), 2 for reply priming
            const count = messages.reduce((acc, m) => {
                return acc + 4 + enc.encode(m.content ?? "").length;
            }, 2);
            enc.free();
            return count;
        }
        enc.free();
        return 0;
    } catch {
        // Fallback for unknown models: rough estimate (1 token ≈ 4 chars)
        const raw = text ?? messages?.map((m) => m.content ?? "").join(" ") ?? "";
        return Math.ceil(raw.length / 4);
    }
}

// ============================================================
// Cost calculation
// ============================================================

export function calculateTotalCostOfLlmFromTranscript(
    messages: Message[],
    costPerInputToken: number,
    costPerOutputToken: number,
    model = "gpt-3.5-turbo",
    checkForCompletion = false,
    endedByAssistant = false,
    completionInputTokenCost = 0.000001,
    completionOutputTokenCost = 0.000002
): [number, LlmTokenUsage] {
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    let completionCheckInputTokens = 0;
    let completionCheckOutputTokens = 0;

    const completionModel = process.env.CHECK_FOR_COMPLETION_LLM ?? model;
    const completionWrongAnswerTokens = countTokens(model, "{'answer': 'No'}");
    const completionRightAnswerTokens = countTokens(model, "{'answer': 'Yes'}");
    const llmTokenUsage: LlmTokenUsage = {};

    for (let i = 0; i < messages.length; i++) {
        const message = messages[i]!;
        if (message.role !== "assistant") continue;

        totalInputTokens += countTokens(model, undefined, messages.slice(0, i));
        totalOutputTokens += countTokens(model, message.content ?? "");

        const completionCheckPrompt: Message[] = [
            { role: "system", content: CHECK_FOR_COMPLETION_PROMPT },
            { role: "user", content: formatMessages(messages.slice(0, i + 1)) },
        ];
        completionCheckInputTokens += countTokens(completionModel, undefined, completionCheckPrompt);

        if (i === messages.length - 1 && endedByAssistant) {
            completionCheckOutputTokens += completionRightAnswerTokens;
        } else {
            completionCheckOutputTokens += completionWrongAnswerTokens;
        }
    }

    const totalCost =
        totalInputTokens * costPerInputToken +
        totalOutputTokens * costPerOutputToken;

    llmTokenUsage[model] = { input: totalInputTokens, output: totalOutputTokens };

    if (checkForCompletion) {
        if (!(completionModel in llmTokenUsage)) {
            llmTokenUsage[completionModel] = { input: 0, output: 0 };
        }
        llmTokenUsage[completionModel]!.input += completionCheckInputTokens;
        llmTokenUsage[completionModel]!.output += completionCheckOutputTokens;

        const completionCost =
            completionCheckInputTokens * completionInputTokenCost +
            completionCheckOutputTokens * completionOutputTokenCost;
        logger.info(`Cost to check completion = ${completionCost}`);

        return [
            Math.round((totalCost + completionCost) * 100000) / 100000,
            llmTokenUsage,
        ];
    }

    return [Math.round(totalCost * 100000) / 100000, llmTokenUsage];
}

// ============================================================
// Extraction details
// ============================================================

export function updateExtractionDetails(
    data: HighLevelAssistantAnalyticsData,
    runDetails: RunDetails
): HighLevelAssistantAnalyticsData | null {
    if (!runDetails.extracted_data) return null;

    const extractionData = runDetails.extracted_data;
    for (const key of Object.keys(extractionData)) {
        const value = extractionData[key]!;
        if (!(key in data.extraction_details)) {
            logger.info(
                `extraction_details: ${JSON.stringify(data.extraction_details)} value: ${value}`
            );
            data.extraction_details[key] = { [value]: 0 };
        } else if (!(value in (data.extraction_details[key] as Record<string, number>))) {
            data.extraction_details[key] = { [value]: 0 };
        }
        (data.extraction_details[key] as Record<string, number>)[value]! += 1;
    }
    return data;
}

// ============================================================
// Execution details
// ============================================================

export function updateExecutionDetails(
    data: HighLevelAssistantAnalyticsData,
    runDetails: RunDetails
): void {
    const totalDuration =
        data.execution_details.average_duration_of_conversation *
        data.execution_details.total_conversations;

    data.execution_details.total_conversations += 1;
    data.execution_details.total_cost += runDetails.total_cost;
    data.execution_details.average_duration_of_conversation =
        (totalDuration + runDetails.conversation_time) /
        data.execution_details.total_conversations;
}

// ============================================================
// Historical values
// ============================================================

export function updateHistoricalValues(
    arr: number[],
    currentRunVal: number,
    lastUpdatedAt: string,
    shouldIncrement: boolean,
    multiplier = 0,
    intervalMinutes = 1440
): number[] {
    const now = new Date();
    const lastUpdated = parseISO(lastUpdatedAt);
    const diffMinutes = differenceInMinutes(now, lastUpdated);

    if (!arr.length) {
        return [0, 0, 0, 0, currentRunVal];
    }

    if (diffMinutes < intervalMinutes) {
        if (shouldIncrement) {
            arr[arr.length - 1]! += currentRunVal;
        } else {
            arr[arr.length - 1] =
                Math.round(
                    ((arr[arr.length - 1]! * multiplier + currentRunVal) / (multiplier + 1)) * 100000
                ) / 100000;
        }
    } else {
        const daysMissed = Math.floor(diffMinutes / intervalMinutes) - 1;
        if (daysMissed > 0) {
            const keep = Math.min(arr.length, daysMissed);
            arr = [...arr.slice(-keep), ...Array(keep).fill(0)];
        }
        if (arr.length < 5) {
            arr.push(currentRunVal);
        } else {
            arr.shift();
            arr.push(currentRunVal);
        }
    }

    return arr;
}

// ============================================================
// Historical spread
// ============================================================

export function updateHistoricalSpread(
    data: HighLevelAssistantAnalyticsData,
    runDetails: RunDetails
): void {
    data.historical_spread.number_of_conversations_in_past_5_days =
        updateHistoricalValues(
            data.historical_spread.number_of_conversations_in_past_5_days,
            1,
            data.last_updated_at,
            true
        );

    data.historical_spread.cost_past_5_days = updateHistoricalValues(
        data.historical_spread.cost_past_5_days,
        runDetails.total_cost,
        data.last_updated_at,
        true
    );

    logger.info(
        `Before updating average duration ${JSON.stringify(data.historical_spread)}`
    );

    const lastCount =
        data.historical_spread.number_of_conversations_in_past_5_days[
        data.historical_spread.number_of_conversations_in_past_5_days.length - 1
        ] ?? 0;

    data.historical_spread.average_duration_past_5_days = updateHistoricalValues(
        data.historical_spread.average_duration_past_5_days,
        runDetails.conversation_time,
        data.last_updated_at,
        false,
        lastCount
    );
}

// ============================================================
// Cost details
// ============================================================

export function updateCostDetails(
    data: HighLevelAssistantAnalyticsData,
    runDetails: RunDetails
): void {
    const total = data.execution_details.total_conversations;

    if (runDetails.cost_breakdown.transcriber > 0) {
        data.cost_details.average_transcriber_cost_per_conversation =
            Math.round(
                ((data.cost_details.average_transcriber_cost_per_conversation * (total - 1) +
                    runDetails.cost_breakdown.transcriber) / total) * 100000
            ) / 100000;
    }

    if (runDetails.cost_breakdown.synthesizer > 0) {
        data.cost_details.average_synthesizer_cost_per_conversation =
            Math.round(
                ((data.cost_details.average_synthesizer_cost_per_conversation * (total - 1) +
                    runDetails.cost_breakdown.synthesizer) / total) * 100000
            ) / 100000;
    }

    data.cost_details.average_llm_cost_per_conversation =
        Math.round(
            ((data.cost_details.average_llm_cost_per_conversation * (total - 1) +
                runDetails.cost_breakdown.llm) / total) * 100000
        ) / 100000;
}

// ============================================================
// Conversation details
// ============================================================

export function updateConversationDetails(
    data: HighLevelAssistantAnalyticsData,
    conversationStatus: "finished" | "rejected" = "finished"
): void {
    data.conversation_details.total_conversations += 1;
    if (conversationStatus === "finished") {
        data.conversation_details.finished_conversations += 1;
    } else {
        data.conversation_details.rejected_conversations += 1;
    }
}

// ============================================================
// Top-level update
// ============================================================

export function updateHighLevelAssistantAnalyticsData(
    current: HighLevelAssistantAnalyticsData | null,
    runDetails: RunDetails
): HighLevelAssistantAnalyticsData {
    logger.info(`run details ${JSON.stringify(runDetails)}`);

    const data: HighLevelAssistantAnalyticsData = current
        ? structuredClone(current)
        : structuredClone(HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA);

    updateExecutionDetails(data, runDetails);
    updateExtractionDetails(data, runDetails);
    updateHistoricalSpread(data, runDetails);
    updateCostDetails(data, runDetails);
    updateConversationDetails(data);

    logger.info(`updated analytics data ${JSON.stringify(data)}`);
    return data;
}