import winston from "winston";
import { AsyncLocalStorage } from "async_hooks";

// ============================================================
// Context Storage (replaces Python's ContextVar)
// ============================================================

type LogContext = Record<string, string>;

const asyncLocalStorage = new AsyncLocalStorage<LogContext>();

export function setLogContext(key: string, value: string): void {
    const ctx = asyncLocalStorage.getStore() ?? {};
    asyncLocalStorage.enterWith({ ...ctx, [key]: value });
}

export function getLogContext(key: string): string {
    return asyncLocalStorage.getStore()?.[key] ?? "";
}

export function clearLogContext(): void {
    asyncLocalStorage.enterWith({});
}

// ============================================================
// Context Format (injected into every log line)
// ============================================================

const contextFormat = winston.format((info) => {
    const ctx = asyncLocalStorage.getStore() ?? {};
    const contextStr = Object.entries(ctx)
        .map(([k, v]) => `{${k}=${v}}`)
        .join(" ");
    info.context = contextStr;
    return info;
});

// ============================================================
// Logger Factory
// ============================================================

type LogLevel = "debug" | "info" | "warn" | "error";

const VALID_LEVELS: LogLevel[] = ["debug", "info", "warn", "error"];

export function configureLogger(moduleName: string, loggingLevel: LogLevel = "info"): winston.Logger {
    const level = VALID_LEVELS.includes(loggingLevel) ? loggingLevel : "info";

    return winston.createLogger({
        level,
        format: winston.format.combine(
            contextFormat(),
            winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss.SSS" }),
            winston.format.printf(({ timestamp, level, message, context }) => {
                const ctx = context ? `${context} ` : "";
                return `${timestamp} ${level.toUpperCase()} ${ctx}{${moduleName}} ${message}`;
            })
        ),
        transports: [new winston.transports.Console()],
    });
}