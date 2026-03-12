import { configureLogger } from "../helper/logger";
import { BaseAgent } from "./base";
import { updatePromptWithContext, getMd5Hash } from "../helper/utils";
import type { BaseLLM } from "../llms/llm";

const logger = configureLogger("graphBasedConversationAgent");

// ============================================================
// Node
// ============================================================

interface AudioTextPair {
    text: string;
    audio?: string;
    [key: string]: unknown;
}

export class Node {
    nodeId: string;
    nodeLabel: string;
    content: AudioTextPair[];
    children: Node[];
    classificationLabels: string[];
    prompt: string | null;
    needResponseGeneration = false;
    milestoneCheckPrompt: string;

    constructor(opts: {
        nodeId: string;
        nodeLabel: string;
        content: AudioTextPair[];
        classificationLabels?: string[];
        prompt?: string | null;
        milestoneCheckPrompt?: string;
        children?: Node[];
    }) {
        this.nodeId = opts.nodeId;
        this.nodeLabel = opts.nodeLabel;
        this.content = opts.content;
        this.children = opts.children ?? [];
        this.classificationLabels = opts.classificationLabels ?? [];
        this.prompt = opts.prompt ?? null;
        this.milestoneCheckPrompt = opts.milestoneCheckPrompt ?? "";
    }
}

// ============================================================
// Graph
// ============================================================

export class Graph {
    preprocessed: boolean;
    root: Node | null = null;
    graph: Map<string, Node>;

    constructor(
        conversationData: Record<string, Record<string, unknown>>,
        preprocessed = false,
        contextData?: Record<string, unknown> | null
    ) {
        this.preprocessed = preprocessed;
        this.graph = this.createGraph(conversationData, contextData);
    }

    private createGraph(
        data: Record<string, Record<string, unknown>>,
        contextData?: Record<string, unknown> | null
    ): Map<string, Node> {
        logger.info("Creating graph");
        const nodeMap = new Map<string, Node>();

        for (const [nodeId, nodeData] of Object.entries(data)) {
            const rawPrompt = (nodeData["prompt"] as string) ?? "";
            const promptParts = rawPrompt.split("###Examples");
            let prompt = rawPrompt;

            if (promptParts.length === 2) {
                const classificationPrompt = promptParts[0]!;
                const userPrompt = updatePromptWithContext(promptParts[1]!, contextData ?? {});
                prompt = [classificationPrompt, userPrompt].join("###Examples");
            }

            const node = new Node({
                nodeId,
                nodeLabel: nodeData["label"] as string,
                content: (nodeData["content"] as AudioTextPair[]) ?? [],
                classificationLabels: (nodeData["classification_labels"] as string[]) ?? [],
                prompt,
                milestoneCheckPrompt: (nodeData["milestone_check_prompt"] as string) ?? "",
                children: [],
            });

            nodeMap.set(nodeId, node);
            if (nodeData["is_root"]) this.root = node;
        }

        // Wire up children after all nodes are created
        for (const [nodeId, nodeData] of Object.entries(data)) {
            const childrenIds = (nodeData["children"] as string[]) ?? [];
            const node = nodeMap.get(nodeId)!;
            node.children = childrenIds
                .map((childId) => nodeMap.get(childId))
                .filter((n): n is Node => n !== undefined);
        }

        return nodeMap;
    }

    // @TODO complete this function
    removeNode(_parent: Node, _node: Node): void {
        console.log("Not yet implemented");
    }
}

// ============================================================
// GraphBasedConversationAgent
// ============================================================

export class GraphBasedConversationAgent extends BaseAgent {
    private llm: BaseLLM;
    private contextData: Record<string, unknown> | null;
    private preprocessed: boolean;
    private graph: Graph | null = null;
    private currentNode: Node | null = null;
    private currentNodeInterim: Node | null = null;
    private conversationIntroDone = false;

    constructor(
        llm: BaseLLM,
        prompts: Record<string, Record<string, unknown>>,
        contextData: Record<string, unknown> | null = null,
        preprocessed = true
    ) {
        super();
        this.llm = llm;
        this.contextData = contextData;
        this.preprocessed = preprocessed;
    }

    loadPromptsAndCreateGraph(
        prompts: Record<string, Record<string, unknown>>
    ): void {
        this.graph = new Graph(prompts, false, this.contextData);
        this.currentNode = this.graph.root;
        // Handle interim node for interim results
        this.currentNodeInterim = this.graph.root;
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    private getAudioTextPair(node: Node): AudioTextPair {
        const ind = Math.floor(Math.random() * node.content.length);
        const audioPair = { ...node.content[ind]! };
        const contextualText = updatePromptWithContext(
            audioPair["text"] as string,
            this.contextData ?? {}
        );
        if (contextualText !== audioPair["text"]) {
            audioPair["text"] = contextualText;
            audioPair["audio"] = getMd5Hash(contextualText);
        }
        return audioPair;
    }

    private async *getNextFormulaic(
        _history: Record<string, unknown>[],
        _stream = true,
        _synthesize = false
    ): AsyncGenerator<AudioTextPair | string> {
        // @TODO add non-preprocessed flow
    }

    private handleIntroMessage(): AudioTextPair {
        const audioPair = this.getAudioTextPair(this.currentNode!);
        this.conversationIntroDone = true;
        logger.info("Conversation intro done");

        // Two-step intros: no prompt on root node
        if (this.currentNode!.prompt === null && this.currentNode!.children.length) {
            const ind = Math.floor(Math.random() * this.currentNode!.children.length);
            this.currentNode = this.currentNode!.children[ind]!;
        }

        return audioPair;
    }

    private async getNextPreprocessedStep(
        history: Record<string, unknown>[]
    ): Promise<AudioTextPair | undefined> {
        logger.info(
            `current node ${this.currentNode!.nodeLabel}, isRoot: ${this.currentNode === this.graph!.root}, introDone: ${this.conversationIntroDone}`
        );

        if (this.currentNode === this.graph!.root && !this.conversationIntroDone) {
            return this.handleIntroMessage();
        }

        logger.info("Conversation intro was done and hence moving forward");

        const prevMessages =
            history.length > 7 ? history.slice(-6) : history.slice(1);

        const message = [
            { role: "system", content: this.currentNode!.prompt },
            ...prevMessages,
        ];

        const response = (await this.llm.generate(message as never, false, false)) as string;
        logger.info(`Classification response ${response}`);

        const classificationResult = JSON.parse(response) as Record<string, unknown>;
        const label = (classificationResult["classification_label"] as string) ?? "";

        for (const child of this.currentNode!.children) {
            if (child.nodeLabel.trim().toLowerCase() === label.trim().toLowerCase()) {
                this.currentNodeInterim = child;
                return this.getAudioTextPair(child);
            }
        }
    }

    updateCurrentNode(): void {
        this.currentNode = this.currentNodeInterim;
    }

    // ------------------------------------------------------------------
    // Generate
    // ------------------------------------------------------------------

    // Label flow is not being used right now as we're logging every request
    async *generate(
        history: Record<string, unknown>[],
        _labelFlow?: unknown
    ): AsyncGenerator<AudioTextPair | string> {
        try {
            if (this.preprocessed) {
                logger.info(`Current node ${this.currentNode?.nodeLabel}`);

                if (!this.currentNode!.children.length) {
                    const ind = Math.floor(Math.random() * this.currentNode!.content.length);
                    const audioPair = this.currentNode!.content[ind]!;
                    logger.info(`Agent: ${audioPair["text"]}`);
                    yield audioPair;
                } else {
                    const nextState = await this.getNextPreprocessedStep(history);
                    logger.info(`Agent: ${JSON.stringify(nextState)}`);
                    if (nextState) yield nextState;
                }

                if (!this.currentNode!.children.length) {
                    await new Promise((resolve) => setTimeout(resolve, 1000));
                    yield "<end_of_conversation>";
                }
            } else {
                yield* this.getNextFormulaic(history as Record<string, unknown>[], true, false);
            }
        } catch (e) {
            logger.error(`Error sending intro text: ${e}`);
            console.error(e);
        }
    }
}