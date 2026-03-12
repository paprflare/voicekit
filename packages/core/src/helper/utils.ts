import fs from "fs";
import fsPromises from "fs/promises";
import path from "path";
import crypto from "crypto";
import { performance } from "perf_hooks";
import {
    S3Client,
    GetObjectCommand,
    PutObjectCommand,
    DeleteObjectCommand,
    ListObjectsCommand,
} from "@aws-sdk/client-s3";
import { configureLogger } from "./logger";
import { format } from "date-fns-tz";
import { PRE_FUNCTION_CALL_MESSAGE, TRANSFERING_CALL_FILLER, PREPROCESS_DIR } from "../constants";
import { DATE_PROMPT } from "../prompts"

const logger = configureLogger("utils");

const BUCKET_NAME = process.env.BUCKET_NAME ?? "";
const RECORDING_BUCKET_NAME = process.env.RECORDING_BUCKET_NAME ?? "";
const RECORDING_BUCKET_URL = process.env.RECORDING_BUCKET_URL ?? "";

// ============================================================
// Types
// ============================================================

export interface MetaInfo {
    request_id?: string;
    sequence_id?: string | null;
    llm_latency?: number | null;
    synthesizer_latency?: number | null;
    transcriber_latency?: number | null;
    is_final?: boolean;
    llm_metadata?: Record<string, unknown>;
    [key: string]: unknown;
}

export interface WsDataPacket {
    data: unknown;
    meta_info: MetaInfo | null;
}

export interface ConversationRecording {
    output: Array<{ start_time: number; duration: number; data: Buffer }>;
    input: { data: Buffer };
    metadata: { started: number };
}

// ============================================================
// DictWithMissing (proxy that returns "" for missing keys)
// ============================================================

export function dictWithMissing(obj: Record<string, string>): Record<string, string> {
    return new Proxy(obj, {
        get(target, key: string) {
            return key in target ? target[key] : "";
        },
    });
}

// ============================================================
// File I/O
// ============================================================

export function loadFile(filePath: string, isJson = false): unknown {
    const raw = fs.readFileSync(filePath, "utf-8");
    return isJson ? JSON.parse(raw) : raw;
}

export function writeJsonFile(filePath: string, data: unknown): void {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 4), "utf-8");
}

// ============================================================
// WebSocket Data Packet
// ============================================================

export function createWsDataPacket(
    data: unknown,
    metaInfo: MetaInfo | null = null,
    isMd5Hash = false,
    llmGenerated = false
): WsDataPacket {
    const metadata = metaInfo ? { ...metaInfo } : null;
    if (metadata !== null) {
        metadata["is_md5_hash"] = isMd5Hash;
        metadata["llm_generated"] = llmGenerated;
    }
    return { data, meta_info: metadata };
}

// ============================================================
// Audio Conversion (using node Buffers — replace with
// a native audio lib like `node-audioworklet` or `sox` as needed)
// ============================================================

/** Int16 PCM -> Float32 normalised to [-1, 1] */
export function int2float(sound: Int16Array): Float32Array {
    const absMax = Math.max(...Array.from(sound).map(Math.abs));
    const floatSound = new Float32Array(sound.length);
    for (let i = 0; i < sound.length; i++) {
        floatSound[i] = absMax > 0 ? sound[i]! / 32768 : 0;
    }
    return floatSound;
}

/** Float32 -> Int16 PCM */
export function float2int(sound: Float32Array): Int16Array {
    const out = new Int16Array(sound.length);
    for (let i = 0; i < sound.length; i++) {
        out[i] = Math.round(sound[i]! * 32767);
    }
    return out;
}

/** Float32 -> clipped Int16 */
export function float32ToInt16(floatAudio: Float32Array): Int16Array {
    const out = new Int16Array(floatAudio.length);
    for (let i = 0; i < floatAudio.length; i++) {
        const clamped = Math.max(-1.0, Math.min(1.0, floatAudio[i]!));
        out[i] = Math.round(clamped * 32767);
    }
    return out;
}

/** μ-law encode a Float32 array */
export function muLawEncode(audio: Float32Array, quantizationChannels = 256): Int32Array {
    const mu = quantizationChannels - 1;
    const out = new Int32Array(audio.length);
    for (let i = 0; i < audio.length; i++) {
        const safeAbs = Math.min(Math.abs(audio[i]!), 1.0);
        const magnitude = Math.log1p(mu * safeAbs) / Math.log1p(mu);
        const signal = Math.sign(audio[i]!) * magnitude;
        out[i] = Math.round(((signal + 1) / 2) * mu + 0.5);
    }
    return out;
}

/** Raw Int16 PCM bytes -> μ-law encoded Int32Array */
export function rawToMulaw(rawBytes: Buffer): Int32Array {
    const samples = new Int16Array(
        rawBytes.buffer,
        rawBytes.byteOffset,
        rawBytes.byteLength / 2
    );
    const floatSamples = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
        floatSamples[i] = samples[i]! / 32768;
    }
    return muLawEncode(floatSamples);
}

/**
 * PCM 16-bit linear -> 8-bit μ-law (G.711).
 * Uses the standard G.711 algorithm (matches Python `audioop.lin2ulaw`).
 */
export function pcmToUlaw(pcmBytes: Buffer): Buffer {
    const samples = new Int16Array(
        pcmBytes.buffer,
        pcmBytes.byteOffset,
        pcmBytes.byteLength / 2
    );
    const ulaw = Buffer.alloc(samples.length);
    for (let i = 0; i < samples.length; i++) {
        let sample = samples[i]!;
        const sign = sample < 0 ? 0x80 : 0;
        if (sample < 0) sample = -sample;
        sample = Math.min(sample + 33, 32767);
        let exp = 7;
        for (let expMask = 0x4000; (sample & expMask) === 0 && exp > 0; expMask >>= 1) exp--;
        const mantissa = (sample >> (exp + 3)) & 0x0f;
        ulaw[i] = ~(sign | (exp << 4) | mantissa) & 0xff;
    }
    return ulaw;
}

// ============================================================
// S3 Helpers
// ============================================================

function makeS3Client(): S3Client {
    return new S3Client({});
}

export async function getS3File(
    bucketName = BUCKET_NAME,
    fileKey = ""
): Promise<Buffer | null> {
    const client = makeS3Client();
    try {
        const response = await client.send(
            new GetObjectCommand({ Bucket: bucketName, Key: fileKey })
        );
        const chunks: Uint8Array[] = [];
        for await (const chunk of response.Body as AsyncIterable<Uint8Array>) {
            chunks.push(chunk);
        }
        return Buffer.concat(chunks);
    } catch (error) {
        logger.error(`getS3File error: ${error}`);
        return null;
    }
}

export async function deleteS3FileByPrefix(
    bucketName: string,
    fileKey: string
): Promise<true | Error> {
    const client = makeS3Client();
    try {
        const listed = await client.send(
            new ListObjectsCommand({ Bucket: bucketName, Prefix: fileKey })
        );
        const tasks = (listed.Contents ?? []).map((file) =>
            client.send(new DeleteObjectCommand({ Bucket: bucketName, Key: file.Key! }))
        );
        await Promise.all(tasks);
        return true;
    } catch (error) {
        logger.error(`deleteS3FileByPrefix error: ${error}`);
        return error as Error;
    }
}

export async function storeFile(opts: {
    bucketName?: string;
    fileKey?: string;
    fileData?: unknown;
    contentType?: string;
    local?: boolean;
    preprocessDir?: string;
}): Promise<void> {
    const {
        bucketName,
        fileKey = "",
        fileData,
        contentType = "json",
        local = false,
        preprocessDir,
    } = opts;

    if (!local) {
        const client = makeS3Client();
        const body =
            contentType === "json"
                ? JSON.stringify(fileData)
                : (fileData as Buffer | string);
        try {
            await client.send(
                new PutObjectCommand({ Bucket: bucketName, Key: fileKey, Body: body as string })
            );
        } catch (error) {
            logger.error(`storeFile S3 error: ${error}`);
        }
        return;
    }

    const dirName = preprocessDir ?? PREPROCESS_DIR;
    const dirPath = path.join(dirName, path.dirname(fileKey));
    fs.mkdirSync(dirPath, { recursive: true });
    try {
        logger.info(`Writing to ${dirName}/${fileKey}`);
        const fullPath = path.join(dirName, fileKey);
        if (contentType === "json") {
            fs.writeFileSync(fullPath, JSON.stringify(fileData), "utf-8");
        } else if (contentType === "csv") {
            fs.writeFileSync(fullPath, fileData as string, "utf-8");
        } else {
            fs.writeFileSync(fullPath, fileData as Buffer);
        }
    } catch (error) {
        logger.error(`Could not save local file: ${error}`);
    }
}

export async function getRawAudioBytes(opts: {
    filename: string;
    agentName?: string;
    audioFormat?: string;
    assistantId?: string;
    local?: boolean;
    isLocation?: boolean;
}): Promise<Buffer | null> {
    const {
        filename,
        agentName,
        audioFormat = "mp3",
        assistantId,
        local = false,
        isLocation = false,
    } = opts;

    if (local) {
        const filePath = isLocation
            ? filename
            : `${PREPROCESS_DIR}/${agentName}/${audioFormat}/${filename}.${audioFormat}`;
        if (!fs.existsSync(filePath)) return null;
        return fs.readFileSync(filePath);
    }

    const objectKey = isLocation
        ? filename
        : `${assistantId}/audio/${filename}.${audioFormat}`;
    logger.info(`Reading ${objectKey}`);
    return getS3File(BUCKET_NAME, objectKey);
}

// ============================================================
// Hashing & Validation
// ============================================================

export function getMd5Hash(text: string): string {
    return crypto.createHash("md5").update(text).digest("hex");
}

export function isValidMd5(hashString: string): boolean {
    return /^[0-9a-f]{32}$/.test(hashString);
}

// ============================================================
// Payload Splitting
// ============================================================

export function splitPayload(
    payload: Buffer,
    maxSize = 500 * 1024
): Buffer | Buffer[] {
    if (payload.length <= maxSize) return payload;
    const chunks: Buffer[] = [];
    for (let i = 0; i < payload.length; i += maxSize) {
        chunks.push(payload.subarray(i, i + maxSize));
    }
    return chunks;
}

// ============================================================
// Task/Toolchain Helpers
// ============================================================

export function getRequiredInputTypes(
    task: Record<string, unknown>
): Record<string, number> {
    const inputTypes: Record<string, number> = {};
    const toolchain = task["toolchain"] as { pipelines: string[][] };
    for (let i = 0; i < toolchain.pipelines.length; i++) {
        const chain = toolchain.pipelines[i]!;
        if (chain[0] === "transcriber") {
            inputTypes["audio"] = i;
        } else if (chain[0] === "synthesizer" || chain[0] === "llm") {
            inputTypes["text"] = i;
        }
    }
    return inputTypes;
}

// ============================================================
// Message Formatting
// ============================================================

export function formatMessages(
    messages: Array<{ role: string; content: string | null }>,
    useSystemPrompt = false,
    includeTools = false
): string {
    let result = "";
    for (const message of messages) {
        if (message.content === null) continue;
        const { role, content } = message;
        if (useSystemPrompt && role === "system") result += `system: ${content}\n`;
        if (role === "assistant") result += `assistant: ${content}\n`;
        else if (role === "user") result += `user: ${content}\n`;
        else if (includeTools && role === "tool") result += `tool_response: ${content}\n`;
    }
    return result;
}

export function updatePromptWithContext(
    prompt: string,
    contextData: Record<string, unknown> | null | undefined
): string {
    try {
        const recipientData =
            contextData && typeof contextData["recipient_data"] === "object"
                ? (contextData["recipient_data"] as Record<string, string>)
                : {};
        const proxy = dictWithMissing(recipientData);
        return prompt.replace(/\{(\w+)\}/g, (_, key) => proxy[key] ?? "");
    } catch {
        return prompt;
    }
}

export async function getPromptResponses(
    assistantId: string,
    local = false
): Promise<unknown | null> {
    const filepath = `${PREPROCESS_DIR}/${assistantId}/conversation_details.json`;
    if (local) {
        logger.info("Loading conversation details from local file");
        try {
            const raw = await fsPromises.readFile(filepath, "utf-8");
            return JSON.parse(raw);
        } catch (error) {
            logger.error(`Could not load dataset: ${error}`);
            return null;
        }
    }

    const key = `${assistantId}/conversation_details.json`;
    logger.info(`Loading conversation details from S3 — bucket: ${BUCKET_NAME}, key: ${key}`);
    try {
        const buf = await getS3File(BUCKET_NAME, key);
        if (!buf) return null;
        return JSON.parse(buf.toString("utf-8"));
    } catch (error) {
        logger.error(`getPromptResponses error: ${error}`);
        return null;
    }
}

export async function executeTasksInChunks(
    tasks: Promise<unknown>[],
    chunkSize = 10
): Promise<void> {
    for (let i = 0; i < tasks.length; i += chunkSize) {
        await Promise.all(tasks.slice(i, i + chunkSize));
    }
}

// ============================================================
// String / JSON Utilities
// ============================================================

export function hasPlaceholders(s: string): boolean {
    return /\{[^{}\s]*\}/.test(s);
}

export function cleanJsonString(jsonStr: unknown): unknown {
    if (typeof jsonStr !== "string") return jsonStr;
    let s = jsonStr;
    if (s.startsWith("```json") && s.endsWith("```")) {
        s = s.slice(7, -3).trim();
    }
    return s.replace("###JSON Structure\n", "");
}

// ============================================================
// PCM / WAV Utilities
// ============================================================

export function* yieldChunksFromMemory(
    audioBytes: Buffer,
    chunkSize = 512
): Generator<Buffer> {
    for (let i = 0; i < audioBytes.length; i += chunkSize) {
        yield audioBytes.subarray(i, i + chunkSize);
    }
}

export function pcmToWavBytes(
    pcmData: Buffer,
    sampleRate = 16000,
    numChannels = 1,
    sampleWidth = 2
): Buffer {
    let data = pcmData;
    if (data.length % 2 === 1) data = Buffer.concat([data, Buffer.alloc(1)]);

    const dataSize = data.length;
    const header = Buffer.alloc(44);
    header.write("RIFF", 0);
    header.writeUInt32LE(36 + dataSize, 4);
    header.write("WAVE", 8);
    header.write("fmt ", 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20); // PCM
    header.writeUInt16LE(numChannels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(sampleRate * numChannels * sampleWidth, 28);
    header.writeUInt16LE(numChannels * sampleWidth, 32);
    header.writeUInt16LE(sampleWidth * 8, 34);
    header.write("data", 36);
    header.writeUInt32LE(dataSize, 40);
    return Buffer.concat([header, data]);
}

export function getSynthAudioFormat(audioBytes: Buffer): "wav" | "pcm" {
    // WAV files start with "RIFF"
    return audioBytes.slice(0, 4).toString("ascii") === "RIFF" ? "wav" : "pcm";
}

export function calculateAudioDuration(
    sizeBytes: number,
    samplingRate: number,
    bitDepth = 16,
    channels = 1,
    format = "wav"
): number {
    const bytesPerSample = format !== "mulaw" ? (bitDepth / 8) * channels : 1;
    const totalSamples = sizeBytes / bytesPerSample;
    return totalSamples / samplingRate;
}

export function createEmptyWavFile(
    durationSeconds: number,
    samplingRate = 24000
): Buffer {
    const totalFrames = Math.round(durationSeconds * samplingRate);
    const silenceSize = totalFrames * 2;
    return pcmToWavBytes(Buffer.alloc(silenceSize), samplingRate);
}

// ============================================================
// Directory Utilities
// ============================================================

export function listNumberOfAudioFilesInDirectory(directory: string): number {
    return fs
        .readdirSync(directory)
        .filter((f) => /\.(mp3|wav|ogg)$/.test(f)).length;
}

export function getFileNamesInDirectory(directory: string): string[] {
    return fs.readdirSync(directory);
}

// ============================================================
// Request Logging
// ============================================================

export async function writeRequestLogs(
    message: Record<string, unknown>,
    runId: string
): Promise<void> {
    const messageData = message["data"] ?? "";
    const row: unknown[] = [
        message["time"],
        message["component"],
        message["direction"],
        message["leg_id"],
        message["sequence_id"],
        message["model"],
    ];

    let componentDetails: unknown[] = [null, null, null, null, null, null, null, null];
    let metadata: Record<string, unknown> = {};

    const component = message["component"] as string;

    if (["llm", "llm_hangup", "llm_voicemail"].includes(component)) {
        const data =
            typeof messageData === "object"
                ? JSON.stringify(messageData)
                : messageData;
        componentDetails = [data, message["input_tokens"] ?? 0, message["output_tokens"] ?? 0, null, message["latency"] ?? null, message["cached"], null, null];
        metadata = (message["llm_metadata"] as Record<string, unknown>) ?? {};
    } else if (component === "transcriber") {
        componentDetails = [messageData, null, null, null, message["latency"] ?? null, false, message["is_final"] ?? false, null];
        metadata = (message["transcriber_metadata"] as Record<string, unknown>) ?? {};
    } else if (component === "synthesizer") {
        const dataStr = String(messageData);
        componentDetails = [dataStr, null, null, dataStr.length, message["latency"] ?? null, message["cached"], null, message["engine"]];
        metadata = (message["synthesizer_metadata"] as Record<string, unknown>) ?? {};
    } else if (["function_call", "graph_routing", "error"].includes(component)) {
        componentDetails = [messageData, null, null, null, message["latency"] ?? null, false, null, null];
        const metaKey = `${component}_metadata`;
        metadata = (message[metaKey] as Record<string, unknown>) ?? {};
    }

    const metadataStr = Object.keys(metadata).length ? JSON.stringify(metadata) : null;
    const fullRow = [...row, ...componentDetails, metadataStr];

    const header =
        "Time,Component,Direction,Leg ID,Sequence ID,Model,Data,Input Tokens,Output Tokens,Characters,Latency,Cached,Final Transcript,Engine,Metadata\n";
    const logLine =
        fullRow
            .map((item) =>
                item !== null && item !== undefined
                    ? `"${String(item).replace(/"/g, '""')}"`
                    : ""
            )
            .join(",") + "\n";

    const logDir = "./logs";
    fs.mkdirSync(logDir, { recursive: true });
    const logFilePath = `${logDir}/${runId}.csv`;
    const fileExists = fs.existsSync(logFilePath);

    const handle = await fsPromises.open(logFilePath, "a");
    await handle.write(fileExists ? logLine : header + logLine);
    await handle.close();
}

export function convertToRequestLog(opts: {
    message: unknown;
    metaInfo: MetaInfo;
    model: string;
    component?: string;
    direction?: string;
    isCached?: boolean;
    engine?: string | null;
    runId?: string;
}): void {
    const {
        message,
        metaInfo,
        model,
        component = "transcriber",
        direction = "response",
        isCached = false,
        engine = null,
        runId,
    } = opts;

    const log: Record<string, unknown> = {
        direction,
        data: message,
        leg_id: metaInfo["request_id"] ?? "-",
        time: new Date().toISOString().replace("T", " ").replace("Z", ""),
        component,
        sequence_id: metaInfo["sequence_id"] ?? null,
        model,
        cached: isCached,
        is_final: false,
        engine,
    };

    if (component === "llm") {
        log["latency"] = direction === "response" ? (metaInfo["llm_latency"] ?? null) : null;
        log["llm_metadata"] = metaInfo["llm_metadata"] ?? null;
    } else if (component === "synthesizer") {
        log["latency"] = direction === "response" ? (metaInfo["synthesizer_latency"] ?? null) : null;
    } else if (component === "transcriber") {
        log["latency"] = direction === "response" ? (metaInfo["transcriber_latency"] ?? null) : null;
        if (metaInfo["is_final"]) log["is_final"] = true;
    } else if (["function_call", "graph_routing"].includes(component)) {
        log["latency"] = null;
        if (component === "graph_routing") {
            log["graph_routing_metadata"] = metaInfo["llm_metadata"] ?? {};
        }
    }

    if (runId) {
        writeRequestLogs(log, runId).catch((e) =>
            logger.error(`writeRequestLogs error: ${e}`)
        );
    }
}

// ============================================================
// Task Cancellation
// ============================================================

export async function processTaskCancellation(task: Promise<void>, taskName: string): Promise<void> {
    if (task instanceof AbortController) {
        task.abort();
    } else {
        const abortController = new AbortController();
        task.then(() => {
            // task completed successfully
        }).catch((error) => {
            // task failed
            abortController.abort();
        });
        await new Promise((resolve, reject) => {
            abortController.signal.addEventListener('abort', resolve);
            abortController.signal.addEventListener('error', reject);
        });
    }
    logger.info(`${taskName} has been successfully cancelled.`);
}

// ============================================================
// Date / Time
// ============================================================

export function getDateTimeFromTimezone(timezone: string): [string, string] {
    const now = new Date();
    const dt = format(now, "EEEE, MMMM dd, yyyy", { timeZone: timezone });
    const ts = format(now, "hh:mm:ss aa", { timeZone: timezone });
    return [dt, ts];
}

// ============================================================
// Language / Message Selection
// ============================================================

export function selectMessageByLanguage(
    messageConfig: string | Record<string, string>,
    detectedLanguage?: string | null
): string {
    if (typeof messageConfig === "string") return messageConfig;
    if (typeof messageConfig === "object") {
        return (
            (detectedLanguage ? messageConfig[detectedLanguage] : undefined) ??
            messageConfig["en"] ??
            Object.values(messageConfig)[0] ??
            ""
        );
    }
    return "";
}

export function hasNonEnglishVariants(
    messageConfig: string | Record<string, string>
): boolean {
    if (typeof messageConfig !== "object") return false;
    const keys = Object.keys(messageConfig);
    return keys.length > 0 && (keys.length > 1 || !("en" in messageConfig));
}

export function computeFunctionPreCallMessage(
    language: string | null,
    functionName: string | null,
    apiToolPreCallMessage: string | Record<string, string> | null
): string {
    const defaultMessage =
        functionName?.startsWith("transfer_call")
            ? TRANSFERING_CALL_FILLER
            : PRE_FUNCTION_CALL_MESSAGE;
    const messageConfig = apiToolPreCallMessage ?? defaultMessage;
    return selectMessageByLanguage(messageConfig, language);
}

// ============================================================
// Timing
// ============================================================

export function nowMs(): number {
    return performance.now();
}

export function timestampMs(): number {
    return Date.now();
}

// ============================================================
// Prompt Structuring
// ============================================================

export function structureSystemPrompt(opts: {
    systemPrompt: string;
    runId: string;
    assistantId: string;
    callSid?: string | null;
    contextData?: Record<string, unknown> | null;
    timezone: string;
    isWebBasedCall?: boolean;
}): string {
    const {
        systemPrompt,
        runId,
        assistantId,
        callSid,
        contextData,
        timezone,
        isWebBasedCall = false,
    } = opts;

    let finalPrompt = systemPrompt;

    const defaultVariables: Record<string, string | undefined> = {
        agent_id: assistantId,
        execution_id: runId,
    };

    if (contextData !== null && contextData !== undefined) {
        const recipientData = (contextData["recipient_data"] ?? {}) as Record<string, string>;
        defaultVariables["agent_number"] = recipientData["agent_number"];
        defaultVariables["user_number"] = recipientData["user_number"];

        if (!isWebBasedCall) {
            finalPrompt = updatePromptWithContext(systemPrompt, contextData);
        }

        if (callSid) defaultVariables["call_sid"] = callSid;

        finalPrompt += "\n\n## Call information:\n\n### Variables:\n";
        for (const [k, v] of Object.entries(defaultVariables)) {
            if (v) finalPrompt += `${k} is "${v}"\n`;
        }
    }

    const [currentDate, currentTime] = getDateTimeFromTimezone(timezone);
    finalPrompt += `\n${DATE_PROMPT(currentDate, currentTime, timezone)}`;
    return finalPrompt;
}


import ps from "alawmulaw";

const { mulaw } = ps

export function ulaw2lin(ulawBuffer: Buffer): Buffer {
    return Buffer.from(mulaw.decode(ulawBuffer));
}

export function lin2ulaw(pcmBuffer: Buffer): Buffer {
    const pcmArray = Int16Array.from(pcmBuffer);
    return Buffer.from(mulaw.encode(pcmArray));
}

export interface TurnLatency {
    turn_id: string | number | null;
    sequence_id: string | number | null;
    first_result_latency_ms: number;
    total_stream_duration_ms: number;
}


/**
 * Strip the 44-byte WAV header and return raw PCM bytes.
 * Assumes standard PCM WAV with a 44-byte header.
 */
export function wavBytesToPcm(wavBuffer: Buffer): Buffer {
    return wavBuffer.subarray(44);
}

/**
 * Resample audio to a target sample rate using fluent-ffmpeg.
 * Supports wav, pcm (s16le), mulaw (mulaw), alaw (alaw) formats.
 */
export async function resample(
    audioBuffer: Buffer,
    targetSampleRate: number,
    opts: {
        format?: string;
        originalSampleRate?: number;
        channels?: number;
    } = {}
): Promise<Buffer> {
    const ffmpeg = (await import("fluent-ffmpeg")).default;
    const { Readable, PassThrough } = await import("stream");

    const format = (opts.format ?? "wav").toLowerCase();
    const channels = opts.channels ?? 1;

    // Map format name to ffmpeg codec/format strings
    const formatMap: Record<string, { inputFormat: string; codec: string }> = {
        wav: { inputFormat: "wav", codec: "pcm_s16le" },
        pcm: { inputFormat: "s16le", codec: "pcm_s16le" },
        mulaw: { inputFormat: "mulaw", codec: "pcm_mulaw" },
        ulaw: { inputFormat: "mulaw", codec: "pcm_mulaw" },
        alaw: { inputFormat: "alaw", codec: "pcm_alaw" },
    };

    const { inputFormat, codec } = formatMap[format] ?? { inputFormat: "wav", codec: "pcm_s16le" };

    return new Promise<Buffer>((resolve, reject) => {
        const chunks: Buffer[] = [];
        const input = new Readable({ read() { } });
        const output = new PassThrough();

        const command = ffmpeg(input)
            .inputFormat(inputFormat)
            .audioFrequency(targetSampleRate)
            .audioChannels(channels)
            .audioCodec(codec)
            .format("wav");

        // Override input sample rate if known
        if (opts.originalSampleRate) {
            command.inputOptions(`-ar ${opts.originalSampleRate}`);
        }

        command
            .on("error", reject)
            .pipe(output);

        output
            .on("data", (chunk: Buffer) => chunks.push(chunk))
            .on("end", () => resolve(Buffer.concat(chunks)))
            .on("error", reject);

        input.push(audioBuffer);
        input.push(null);
    });
}

/**
 * Convert audio buffer (mp3, ogg, etc.) to WAV using fluent-ffmpeg.
 */
export async function convertAudioToWav(
    audioBuffer: Buffer,
    inputFormat: string
): Promise<Buffer> {
    const ffmpeg = (await import("fluent-ffmpeg")).default;
    const { Readable, PassThrough } = await import("stream");

    return new Promise<Buffer>((resolve, reject) => {
        const chunks: Buffer[] = [];
        const input = new Readable({ read() { } });
        const output = new PassThrough();

        ffmpeg(input)
            .inputFormat(inputFormat)
            .audioFrequency(8000)
            .audioChannels(1)
            .format("wav")
            .on("error", reject)
            .pipe(output);

        output
            .on("data", (chunk: Buffer) => chunks.push(chunk))
            .on("end", () => resolve(Buffer.concat(chunks)))
            .on("error", reject);

        input.push(audioBuffer);
        input.push(null);
    });
}


import ffmpeg from "fluent-ffmpeg";
import { Readable, PassThrough } from "stream";

// ============================================================
// saveAudioFileToS3
// Replaces Python: torchaudio.load → resample → torchaudio.save → S3 upload
// ============================================================

/**
 * Resample raw PCM/WAV audio to a target sample rate via ffmpeg,
 * encode as WAV, and upload to the recording S3 bucket.
 *
 * Mirrors Python:
 *   waveform, sample_rate = torchaudio.load(BytesIO(audio_data))
 *   resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
 *   resampled = resampler(waveform)
 *   torchaudio.save(buf, resampled, target_sample_rate)
 *   s3.put_object(Bucket=RECORDING_BUCKET_NAME, Key=file_name, Body=buf.getvalue())
 *
 * @param audioData   Raw audio bytes (PCM or WAV)
 * @param fileName    S3 object key to write to
 * @param inputSampleRate   Original sample rate of audioData (default 8000)
 * @param targetSampleRate  Desired output sample rate (default 8000)
 * @param inputFormat       ffmpeg input format hint: "wav", "s16le", "mulaw", etc.
 */
export async function saveAudioFileToS3(
    audioData: Buffer,
    fileName: string,
    inputSampleRate = 8000,
    targetSampleRate = 8000,
    inputFormat = "s16le"
): Promise<void> {
    try {
        // ------------------------------------------------------------------
        // Step 1: Resample + encode to WAV using ffmpeg (replaces torchaudio)
        // ------------------------------------------------------------------
        const wavBuffer = await new Promise<Buffer>((resolve, reject) => {
            const chunks: Buffer[] = [];

            // Feed raw bytes into ffmpeg via a Readable stream
            const inputStream = new Readable({ read() { } });

            const passthrough = new PassThrough();
            passthrough.on("data", (chunk: Buffer) => chunks.push(chunk));
            passthrough.on("end", () => resolve(Buffer.concat(chunks)));
            passthrough.on("error", reject);

            ffmpeg(inputStream)
                .inputFormat(inputFormat)
                .inputOptions([`-ar ${inputSampleRate}`, "-ac 1"])
                .audioFrequency(targetSampleRate)
                .audioChannels(1)
                .audioCodec("pcm_s16le")
                .format("wav")
                .on("error", reject)
                .pipe(passthrough, { end: true });

            // Push data then signal EOF
            inputStream.push(audioData);
            inputStream.push(null);
        });

        // ------------------------------------------------------------------
        // Step 2: Upload WAV to S3 (replaces boto3 s3.put_object)
        // ------------------------------------------------------------------
        const client = new S3Client({});
        await client.send(
            new PutObjectCommand({
                Bucket: RECORDING_BUCKET_NAME,
                Key: fileName,
                Body: wavBuffer,
                ContentType: "audio/wav",
            })
        );

        logger.info(`saveAudioFileToS3: uploaded ${fileName} (${wavBuffer.length} bytes) to ${RECORDING_BUCKET_NAME}`);
    } catch (e) {
        logger.error(`saveAudioFileToS3 error: ${e}`);
    }
}