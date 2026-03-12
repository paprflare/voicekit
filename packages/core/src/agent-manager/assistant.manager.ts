import { v4 as uuidv4 } from "uuid";
import { BaseManager } from "./base";
import { TaskManager } from "./taskManager";
import { configureLogger } from "../helper/logger";
import { AGENT_WELCOME_MESSAGE } from "../models";
import { updatePromptWithContext } from "../helper/utils";

const logger = configureLogger("assistantManager");

export class AssistantManager extends BaseManager {
    private runId: string;
    private assistantId: string | null;
    private tools: Record<string, unknown>;
    private websocket: unknown;
    private agentConfig: Record<string, unknown>;
    private contextData: Record<string, unknown> | null;
    private tasks: Record<string, unknown>[];
    private taskStates: boolean[];
    private turnBasedConversation: boolean | null;
    private cache: unknown;
    private inputQueue: unknown;
    private outputQueue: unknown;
    private kwargs: Record<string, unknown>;
    private conversationHistory: unknown;

    constructor(opts: {
        agentConfig: Record<string, unknown>;
        ws?: unknown;
        assistantId?: string | null;
        contextData?: Record<string, unknown> | null;
        conversationHistory?: unknown;
        turnBasedConversation?: boolean | null;
        cache?: unknown;
        inputQueue?: unknown;
        outputQueue?: unknown;
        [key: string]: unknown;
    }) {
        super();

        this.runId = uuidv4();
        this.assistantId = opts.assistantId ?? null;
        this.tools = {};
        this.websocket = opts.ws ?? null;
        this.agentConfig = opts.agentConfig;
        this.contextData = opts.contextData ?? null;
        this.tasks = (opts.agentConfig["tasks"] as Record<string, unknown>[]) ?? [];
        this.taskStates = new Array(this.tasks.length).fill(false);
        this.turnBasedConversation = opts.turnBasedConversation ?? null;
        this.cache = opts.cache ?? null;
        this.inputQueue = opts.inputQueue ?? null;
        this.outputQueue = opts.outputQueue ?? null;
        this.conversationHistory = opts.conversationHistory ?? null;

        // Strip known keys; remainder are kwargs
        const { agentConfig, ws, assistantId, contextData, conversationHistory,
            turnBasedConversation, cache, inputQueue, outputQueue, ...rest } = opts;
        this.kwargs = rest;

        const welcomeMsg = (opts.agentConfig["agent_welcome_message"] as string | undefined) ?? AGENT_WELCOME_MESSAGE;
        if (opts["is_web_based_call"]) {
            this.kwargs["agent_welcome_message"] = welcomeMsg;
        } else {
            this.kwargs["agent_welcome_message"] = updatePromptWithContext(welcomeMsg, this.contextData);
        }
    }

    async *run(opts?: {
        local?: boolean;
        runId?: string;
    }): AsyncGenerator<[number, Record<string, unknown>]> {
        const local = opts?.local ?? false;

        if (opts?.runId) {
            this.runId = opts.runId;
        }

        const agentName =
            (this.agentConfig["agent_name"] as string | undefined) ??
            (this.agentConfig["assistant_name"] as string | undefined);

        let inputParameters: Record<string, unknown> | null = null;

        for (let taskId = 0; taskId < this.tasks.length; taskId++) {
            const task = this.tasks[taskId]!;
            logger.info(`Running task ${taskId}`);

            const taskManager: TaskManager = new TaskManager({
                agentName,
                taskId,
                task,
                websocket: this.websocket,
                contextData: this.contextData,
                inputParameters,
                assistantId: this.assistantId,
                runId: this.runId,
                turnBasedConversation: this.turnBasedConversation,
                cache: this.cache,
                inputQueue: this.inputQueue,
                outputQueue: this.outputQueue,
                conversationHistory: this.conversationHistory,
                ...this.kwargs,
            });

            await taskManager.loadPrompt(agentName, taskId, { local, ...this.kwargs });

            const taskOutput = await taskManager.run();
            taskOutput["run_id"] = this.runId;

            yield [taskId, structuredClone(taskOutput)];

            this.taskStates[taskId] = true;

            if (taskId === 0) {
                inputParameters = taskOutput;
            }

            if (task["task_type"] === "extraction") {
                inputParameters = inputParameters ?? {};
                inputParameters["extraction_details"] = taskOutput["extracted_data"];
            }
        }

        logger.info("Done with execution of the agent");
    }
}