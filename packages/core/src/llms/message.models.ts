import { z } from "zod";
import { ChatRoleSchema, ResponseItemTypeSchema } from "../enums";

// ============================================================
// Schemas
// ============================================================

export const ChatToolCallFunctionSchema = z.object({
    name: z.string().default(""),
    arguments: z.string().default(""),
});

export const ChatToolCallSchema = z.object({
    id: z.string().default(""),
    function: ChatToolCallFunctionSchema.default({
        name: "",
        arguments: "",
    }),
});

export const ChatMessageSchema = z.object({
    role: z.string(),
    content: z.string().nullable().optional(),
    tool_calls: z.array(ChatToolCallSchema).nullable().optional(),
    tool_call_id: z.string().nullable().optional(),
});

export const ChatToolFunctionSchema = z.object({
    name: z.string().default(""),
    description: z.string().default(""),
    parameters: z.record(z.any(), z.any()).default({}),
    strict: z.boolean().default(false),
});

export const ChatToolDefinitionSchema = z.object({
    type: z.string().default("function"),
    function: ChatToolFunctionSchema.default({
        name: "",
        description: "",
        parameters: {},
        strict: false,
    }),
});

// ============================================================
// Types
// ============================================================

export type ChatToolCallFunction = z.infer<typeof ChatToolCallFunctionSchema>;
export type ChatToolCall = z.infer<typeof ChatToolCallSchema>;
export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type ChatToolFunction = z.infer<typeof ChatToolFunctionSchema>;
export type ChatToolDefinition = z.infer<typeof ChatToolDefinitionSchema>;

// ============================================================
// MessageFormatAdapter
// ============================================================

export class MessageFormatAdapter {
    /** Chat Completions messages -> [instructions, Responses API input items] */
    static chatToResponsesInput(messages: Record<string, unknown>[]): [string, Record<string, unknown>[]] {
        let instructions = "";
        const inputItems: Record<string, unknown>[] = [];

        const parsed = messages.map((msg) => ChatMessageSchema.parse(msg));

        for (const msg of parsed) {
            if (msg.role === ChatRoleSchema.enum.system) {
                instructions = msg.content ?? "";

            } else if (msg.role === ChatRoleSchema.enum.user) {
                inputItems.push({
                    type: ResponseItemTypeSchema.enum.message,
                    role: ChatRoleSchema.enum.user,
                    content: msg.content ?? "",
                });

            } else if (msg.role === ChatRoleSchema.enum.assistant) {
                if (msg.tool_calls?.length) {
                    for (const tc of msg.tool_calls) {
                        inputItems.push({
                            type: ResponseItemTypeSchema.enum.function_call,
                            call_id: tc.id,
                            name: tc.function.name,
                            arguments: tc.function.arguments,
                        });
                    }
                } else if (msg.content != null) {
                    inputItems.push({
                        type: ResponseItemTypeSchema.enum.message,
                        role: ChatRoleSchema.enum.assistant,
                        content: msg.content,
                    });
                }

            } else if (msg.role === ChatRoleSchema.enum.tool) {
                inputItems.push({
                    type: ResponseItemTypeSchema.enum.function_call_output,
                    call_id: msg.tool_call_id ?? "",
                    output: msg.content ?? "",
                });
            }
        }

        return [instructions, inputItems];
    }

    /**
     * Flatten nested tool schema for Responses API.
     *
     * {"type":"function","function":{name,desc,params}}
     * -> {"type":"function","name":...,"description":...,"parameters":...,"strict":true}
     */
    static chatToolsToResponsesTools(chatTools: Record<string, unknown>[]): Record<string, unknown>[] {
        return chatTools.map((raw) => {
            const tool = ChatToolDefinitionSchema.parse(raw);
            return {
                type: ResponseItemTypeSchema.enum.function,
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters,
                strict: tool.function.strict,
            };
        });
    }
}