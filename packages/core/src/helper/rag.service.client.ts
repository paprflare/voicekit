import { configureLogger } from "./logger";
import { v4 as uuidv4 } from "uuid";

const logger = configureLogger("ragServiceClient");

// ============================================================
// Types
// ============================================================

export interface RAGContext {
    text: string;
    score: number;
    metadata: Record<string, unknown>;
}

export interface RAGResponse {
    contexts: RAGContext[];
    total_results: number;
    processing_time: number;
    total_query_time_ms: number;
    server_processing_time_ms: number;
}

function emptyRAGResponse(): RAGResponse {
    return {
        contexts: [],
        total_results: 0,
        processing_time: 0,
        total_query_time_ms: 0,
        server_processing_time_ms: 0,
    };
}

// ============================================================
// RAGServiceClient
// ============================================================

export class RAGServiceClient {
    private baseUrl: string;
    private timeoutMs: number;

    private consecutiveFailures = 0;
    private readonly failureThreshold = 3;
    private lastFailureTime = 0;
    private readonly cooldownSeconds = 30;

    constructor(ragServerUrl: string, timeoutSeconds = 5) {
        this.baseUrl = ragServerUrl.replace(/\/$/, "");
        this.timeoutMs = timeoutSeconds * 1000;
    }

    // ------------------------------------------------------------------
    // Availability guard
    // ------------------------------------------------------------------

    private isAvailable(): boolean {
        if (this.consecutiveFailures < this.failureThreshold) return true;
        if (Date.now() / 1000 - this.lastFailureTime >= this.cooldownSeconds) {
            this.lastFailureTime = Date.now() / 1000;
            return true;
        }
        return false;
    }

    private onFailure(): void {
        this.consecutiveFailures++;
        this.lastFailureTime = Date.now() / 1000;
    }

    private onSuccess(): void {
        if (this.consecutiveFailures > 0) {
            logger.info(`RAG service recovered after ${this.consecutiveFailures} failures`);
        }
        this.consecutiveFailures = 0;
    }

    // ------------------------------------------------------------------
    // Fetch helper (with timeout via AbortController)
    // ------------------------------------------------------------------

    private async fetchWithTimeout(
        url: string,
        options: RequestInit = {}
    ): Promise<Response> {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this.timeoutMs);
        try {
            return await fetch(url, { ...options, signal: controller.signal });
        } finally {
            clearTimeout(timer);
        }
    }

    // ------------------------------------------------------------------
    // Health
    // ------------------------------------------------------------------

    async healthCheck(): Promise<Record<string, unknown>> {
        try {
            const res = await this.fetchWithTimeout(`${this.baseUrl}/`);
            if (res.ok) return (await res.json()) as Record<string, unknown>;
            return { status: "unhealthy", error: `HTTP ${res.status}` };
        } catch (e) {
            logger.error(`Health check failed: ${e}`);
            return { status: "error", error: String(e) };
        }
    }

    async checkCollectionHealth(
        collectionId: string
    ): Promise<Record<string, unknown>> {
        try {
            const res = await this.fetchWithTimeout(
                `${this.baseUrl}/collections/${collectionId}/health`
            );
            return (await res.json()) as Record<string, unknown>;
        } catch (e) {
            logger.error(`Collection health check failed: ${e}`);
            return {
                collection_id: collectionId,
                status: "error",
                accessible: false,
                error: String(e),
            };
        }
    }

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    async queryForConversation(
        query: string,
        collections: string[],
        maxResults = 15,
        similarityThreshold = 0.0
    ): Promise<RAGResponse> {
        const queryId = uuidv4();
        const queryPreview = query.length > 100 ? `${query.slice(0, 100)}...` : query;

        if (!this.isAvailable()) {
            logger.warn(
                `RAG query SKIPPED (service down) | query_id: ${queryId} | consecutive_failures: ${this.consecutiveFailures}`
            );
            return emptyRAGResponse();
        }

        logger.info(
            `RAG query started | query_id: ${queryId} | collections: ${collections} | query: '${queryPreview}' | max_results: ${maxResults}`
        );

        const startTime = Date.now();
        const payload = {
            query,
            collections,
            max_results: maxResults,
            similarity_threshold: similarityThreshold,
        };

        try {
            const res = await this.fetchWithTimeout(
                `${this.baseUrl}/conversation/query`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-Query-ID": queryId,
                    },
                    body: JSON.stringify(payload),
                }
            );

            if (!res.ok) {
                const errorText = await res.text();
                logger.error(
                    `RAG query failed | query_id: ${queryId} | status: ${res.status} | error: ${errorText}`
                );
                this.onFailure();
                return emptyRAGResponse();
            }

            const data = (await res.json()) as Record<string, unknown>;
            const totalQueryTimeMs = Date.now() - startTime;

            const documents = (data["documents"] as Record<string, unknown>[]) ?? [];
            const contexts: RAGContext[] = documents.map((doc) => ({
                text: (doc["text"] as string) ?? "",
                score: (doc["score"] as number) ?? 0.0,
                metadata: (doc["metadata"] as Record<string, unknown>) ?? {},
            }));

            const totalResults = (data["total_retrieved"] as number) ?? contexts.length;
            const serverProcessingTimeMs = Number(data["query_time_ms"] ?? 0);
            const processingTime = serverProcessingTimeMs / 1000;

            logger.info(
                `RAG query completed | query_id: ${queryId} | results: ${totalResults} | server_time: ${processingTime.toFixed(3)}s | total_time: ${totalQueryTimeMs.toFixed(1)}ms`
            );

            this.onSuccess();

            return {
                contexts,
                total_results: totalResults,
                processing_time: processingTime,
                total_query_time_ms: totalQueryTimeMs,
                server_processing_time_ms: serverProcessingTimeMs,
            };
        } catch (e) {
            if (e instanceof Error && e.name === "AbortError") {
                logger.error(
                    `RAG query timeout | query_id: ${queryId} | collections: ${collections}`
                );
            } else {
                logger.error(`RAG query error | query_id: ${queryId} | error: ${e}`);
            }
            this.onFailure();
            return emptyRAGResponse();
        }
    }

    // ------------------------------------------------------------------
    // Prompt helpers
    // ------------------------------------------------------------------

    formatContextForPrompt(contexts: RAGContext[]): string {
        if (!contexts.length) return "";
        return contexts
            .map(
                (ctx, i) =>
                    `Context ${i + 1} (relevance: ${ctx.score.toFixed(3)}):\n${ctx.text}\n`
            )
            .join("\n");
    }

    async getEnhancedPrompt(
        originalPrompt: string,
        userQuery: string,
        collections: string[],
        maxResults = 5
    ): Promise<string> {
        const ragResponse = await this.queryForConversation(
            userQuery,
            collections,
            maxResults
        );

        if (!ragResponse.contexts.length) return originalPrompt;

        const contextText = this.formatContextForPrompt(ragResponse.contexts);

        return `${originalPrompt}

RELEVANT CONTEXT:
${contextText}
Please respond to the user's query using the above context when relevant. If the context doesn't contain relevant information, respond based on your general knowledge.`;
    }

    async close(): Promise<void> {
        // fetch-based client has no persistent session to close
    }
}

// ============================================================
// Singleton
// ============================================================

export class RAGServiceClientSingleton {
    private static instance: RAGServiceClient | null = null;
    private static url: string | null = null;

    static getClient(ragServerUrl: string): RAGServiceClient {
        if (!this.instance || this.url !== ragServerUrl) {
            this.instance = new RAGServiceClient(ragServerUrl);
            this.url = ragServerUrl;
        }
        return this.instance;
    }

    static async closeClient(): Promise<void> {
        if (this.instance) {
            await this.instance.close();
            this.instance = null;
            this.url = null;
        }
    }
}