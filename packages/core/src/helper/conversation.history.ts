import { ChatRoleSchema } from "../enums";

const UNHEARD_ROLES = new Set([
    ChatRoleSchema.enum.assistant,
    ChatRoleSchema.enum.tool,
]);

export interface Message {
    role: string;
    content: string | null;
    tool_calls?: unknown[];
    tool_call_id?: string;
}

export class ConversationHistory {
    private _messages: Message[];
    private _interim: Message[];

    constructor(initialHistory: Message[] | null = null) {
        this._messages = initialHistory ?? [];
        this._interim = structuredClone(this._messages);
    }

    // ------------------------------------------------------------------
    // Setup
    // ------------------------------------------------------------------

    setupSystemPrompt(systemPrompt: Message, welcomeMessage = ""): void {
        if (systemPrompt.content) {
            this._messages = this._messages.length
                ? [systemPrompt, ...this._messages]
                : [systemPrompt];
        }

        if (welcomeMessage && this._messages.length === 1) {
            this._messages.push({
                role: ChatRoleSchema.enum.assistant,
                content: welcomeMessage,
            });
        }

        this._interim = structuredClone(this._messages);
    }

    // ------------------------------------------------------------------
    // Append
    // ------------------------------------------------------------------

    appendUser(content: string): void {
        this._messages.push({ role: ChatRoleSchema.enum.user, content });
    }

    appendAssistant(content: string, toolCalls?: unknown[]): void {
        const msg: Message = { role: ChatRoleSchema.enum.assistant, content };
        if (toolCalls !== undefined) msg.tool_calls = toolCalls;
        this._messages.push(msg);
    }

    appendToolResult(toolCallId: string, content: string): void {
        this._messages.push({
            role: ChatRoleSchema.enum.tool,
            tool_call_id: toolCallId,
            content,
        });
    }

    // ------------------------------------------------------------------
    // Update
    // ------------------------------------------------------------------

    updateSystemPrompt(content: string): void {
        if (this._messages[0]?.role === ChatRoleSchema.enum.system) {
            this._messages[0].content = content;
        }
    }

    updateWelcomeMessage(content: string): void {
        if (this._messages[1]?.role === ChatRoleSchema.enum.assistant) {
            this._messages[1].content = content;
        }
    }

    // ------------------------------------------------------------------
    // Pop / merge
    // ------------------------------------------------------------------

    popUnheardResponses(): Message[] {
        const popped: Message[] = [];
        while (
            this._messages.length &&
            UNHEARD_ROLES.has(this._messages[this._messages.length - 1]!.role as 'assistant' | 'tool')
        ) {
            popped.push(this._messages.pop()!);
        }
        return popped;
    }

    popAndMergeUser(newContent: string): string {
        if (this._messages.at(-1)?.role === ChatRoleSchema.enum.user) {
            const prev = this._messages.pop()!;
            return `${prev.content} ${newContent}`;
        }
        return newContent;
    }

    // ------------------------------------------------------------------
    // Interruption sync
    // ------------------------------------------------------------------

    syncAfterInterruption(
        responseHeard: string | null,
        updateFn: (original: string, heard: string | null) => string
    ): void {
        ConversationHistory._trimLastAssistant(this._messages, responseHeard, updateFn);
    }

    syncInterimAfterInterruption(
        responseHeard: string | null,
        updateFn: (original: string, heard: string | null) => string
    ): void {
        ConversationHistory._trimLastAssistant(this._interim, responseHeard, updateFn);
    }

    private static _trimLastAssistant(
        msgs: Message[],
        responseHeard: string | null,
        updateFn: (original: string, heard: string | null) => string
    ): void {
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i]!.role === ChatRoleSchema.enum.assistant) {
                const original = msgs[i]!.content;
                if (original === null) continue;
                const updated = updateFn(original, responseHeard);
                if (!updated?.trim()) {
                    msgs.splice(i, 1);
                } else {
                    msgs[i]!.content = updated;
                }
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // Interim sync
    // ------------------------------------------------------------------

    syncInterim(messages: Message[] | null = null): void {
        this._interim = structuredClone(messages ?? this._messages);
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    get messages(): Message[] {
        return this._messages;
    }

    get interim(): Message[] {
        return this._interim;
    }

    get lastRole(): string | null {
        return this._messages.at(-1)?.role ?? null;
    }

    get lastContent(): string | null {
        return this._messages.at(-1)?.content ?? null;
    }

    getCopy(): Message[] {
        return structuredClone(this._messages);
    }

    isDuplicateUser(content: string): boolean {
        const last = this._messages.at(-1);
        return (
            last?.role === ChatRoleSchema.enum.user &&
            (last.content ?? "").trim() === content.trim()
        );
    }

    get length(): number {
        return this._messages.length;
    }
}