import { configureLogger } from "./logger";
import { convertToRequestLog } from "./utils";

const logger = configureLogger("apiHelper");

// ============================================================
// $var marker substitution
// ============================================================

function containsVarMarkers(obj: unknown): boolean {
    if (typeof obj === "object" && obj !== null) {
        if (Array.isArray(obj)) {
            return obj.some(containsVarMarkers);
        }
        const record = obj as Record<string, unknown>;
        if ("$var" in record) return true;
        return Object.values(record).some(containsVarMarkers);
    }
    return false;
}

export function substituteVarMarkers(
    obj: unknown,
    values: Record<string, unknown>
): unknown {
    if (typeof obj === "object" && obj !== null) {
        if (Array.isArray(obj)) {
            return obj.map((item) => substituteVarMarkers(item, values));
        }
        const record = obj as Record<string, unknown>;
        // $var marker: { "$var": "name" }
        if (Object.keys(record).length === 1 && "$var" in record) {
            const varName = record["$var"] as string;
            if (varName in values) return values[varName];
            logger.warn(`No value provided for $var marker: ${varName}`);
            return obj;
        }
        return Object.fromEntries(
            Object.entries(record).map(([k, v]) => [k, substituteVarMarkers(v, values)])
        );
    }
    return obj; // primitives returned as-is
}

// ============================================================
// Legacy %(field)s template substitution
// ============================================================

function substituteLegacyTemplate(
    template: string,
    kwargs: Record<string, unknown>
): string {
    // Replaces %(key)s with the value, JSON-serialising objects/arrays
    return template.replace(/%\((\w+)\)s/g, (_, key: string) => {
        const val = kwargs[key];
        if (val === undefined) return "";
        if (typeof val === "object" && val !== null) return JSON.stringify(val);
        return String(val);
    });
}

// ============================================================
// Form normalisation
// ============================================================

export function normalizeForForm(data: Record<string, unknown>): Record<string, string> {
    return Object.fromEntries(
        Object.entries(data).map(([k, v]) => [
            k,
            typeof v === "object" && v !== null ? JSON.stringify(v) : String(v),
        ])
    );
}

// ============================================================
// Main API trigger
// ============================================================

export async function triggerApi(
    url: string,
    method: string,
    param: string | Record<string, unknown> | null,
    apiToken: string | null,
    headersData: Record<string, string> | null,
    metaInfo: Record<string, unknown>,
    runId: string,
    kwargs: Record<string, unknown> = {}
): Promise<string> {
    try {
        let requestBody: string | null = null;
        let apiParams: Record<string, unknown> | null = null;

        if (param) {
            if (typeof param === "object" && containsVarMarkers(param)) {
                // New format: $var marker substitution
                apiParams = substituteVarMarkers(param, kwargs) as Record<string, unknown>;
                requestBody = JSON.stringify(apiParams);
                logger.info("Using $var marker substitution for param");
            } else {
                // Legacy format: %(field)s string template
                const paramStr =
                    typeof param === "object" ? JSON.stringify(param) : param;
                requestBody = substituteLegacyTemplate(paramStr, kwargs);
                apiParams = JSON.parse(requestBody) as Record<string, unknown>;
            }
        }

        const headers: Record<string, string> = { "Content-Type": "application/json" };
        if (apiToken) headers["Authorization"] = apiToken;
        if (headersData) Object.assign(headers, headersData);

        const contentType = headers["Content-Type"]?.toLowerCase().startsWith(
            "application/x-www-form-urlencoded"
        )
            ? "form"
            : "json";

        convertToRequestLog({
            message: requestBody,
            metaInfo: metaInfo as never,
            model: null as never,
            component: "function_call",
            direction: "request",
            isCached: false,
            runId,
        });

        // Replaces asyncio.sleep(0.7)
        await new Promise((resolve) => setTimeout(resolve, 700));

        const lowerMethod = method.toLowerCase();

        if (lowerMethod === "get") {
            const queryUrl = new URL(url);
            if (apiParams) {
                for (const [k, v] of Object.entries(apiParams)) {
                    queryUrl.searchParams.set(k, String(v));
                }
            }
            logger.info(`Sending GET request ${requestBody}, ${url}, ${JSON.stringify(headers)}`);
            const res = await fetch(queryUrl.toString(), { method: "GET", headers });
            return res.text();
        }

        if (lowerMethod === "post") {
            logger.info(`Sending POST request ${JSON.stringify(apiParams)}, ${url}, ${JSON.stringify(headers)}`);

            if (contentType === "form") {
                const formData = new URLSearchParams(
                    normalizeForForm(apiParams ?? {})
                );
                const res = await fetch(url, {
                    method: "POST",
                    headers,
                    body: formData.toString(),
                });
                return res.text();
            }

            const res = await fetch(url, {
                method: "POST",
                headers,
                body: JSON.stringify(apiParams),
            });
            return res.text();
        }

        throw new Error(`Unsupported HTTP method: ${method}`);
    } catch (e) {
        const message = `ERROR CALLING API: Please check your API: ${e}`;
        logger.error(message);
        return message;
    }
}

// ============================================================
// Response parsing
// ============================================================

export async function computedApiResponse(
    response: string
): Promise<[string[] | null, unknown[] | null]> {
    try {
        const parsed = JSON.parse(response) as Record<string, unknown>;
        return [Object.keys(parsed), Object.values(parsed)];
    } catch {
        return [null, null];
    }
}