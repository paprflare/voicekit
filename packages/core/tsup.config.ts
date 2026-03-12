import { defineConfig } from 'tsup'

/**
 * @voice-kit/core — tsup build config
 *
 * Node-only package. Ships ESM + CJS + type declarations.
 * All dependencies are externalized — consumers install them.
 *
 * Shims:
 *   - None. This is a pure Node.js package.
 *
 * Special handling:
 *   - fluent-ffmpeg uses spawn() at runtime — cannot be bundled.
 *   - opusscript ships a native .wasm — excluded from bundle.
 *   - @ricky0123/vad-web ships its own WASM/worker — excluded.
 *   - OpenTelemetry packages have peer-dep graphs — externalized.
 */
export default defineConfig((options) => ({
    entry: {
        'index': 'src/index.ts',
        'agents/index': 'src/agents/index.ts',
        'react/index': 'src/react/index.ts',
        'telephony/index': 'src/telephony/index.ts',
    },

    format: ['esm', 'cjs'],

    // ── Type declarations ────────────────────────────────────────────────────────
    dts: true,
    sourcemap: true,

    // ── Target ───────────────────────────────────────────────────────────────────
    // Node 18+ (native fetch, structuredClone, AsyncIterator helpers)
    target: 'node18',
    platform: 'node',

    // ── Splitting: ESM only ───────────────────────────────────────────────────────
    // CJS cannot benefit from code splitting; ESM splits are useful for tree-shaking
    splitting: false,   // false keeps output predictable for monorepo consumers
    treeshake: true,

    // ── Clean output on every build ───────────────────────────────────────────────
    clean: true,
    outDir: 'dist',

    // ── Externalize everything except src/ ────────────────────────────────────────
    // All runtime dependencies are peer/runtime installs.
    // noExternal would bundle them — bad for native modules (ffmpeg, opusscript).
    noExternal: [],   // Default: bundle nothing from node_modules

    external: [
        // Native / WASM / spawn-based — must never be bundled
        'fluent-ffmpeg',
        'opusscript',
        '@ricky0123/vad-web',
        'node-record-lpcm16',
        'mic',

        // Large SDKs with their own resolution logic
        '@deepgram/sdk',
        'assemblyai',
        'elevenlabs',
        '@cartesia/sdk',
        'livekit-server-sdk',
        'mastra',

        // OTel — has complex peer graph, safe to externalize
        '@opentelemetry/sdk-node',
        '@opentelemetry/sdk-trace-node',
        '@opentelemetry/exporter-trace-otlp-http',
        '@opentelemetry/semantic-conventions',
        '@opentelemetry/resources',
        '@opentelemetry/sdk-trace-base',
        '@opentelemetry/api',

        // AI SDK — large, tree-shakes itself
        'ai',
        '@ai-sdk/openai',

        // Lightweight but consumed by callers too — avoid double-bundling
        'ws',
        'axios',
        'zod',
        'pino',
        'lru-cache',
        'libphonenumber-js',
        'i18next',
        'sarvam-ai',
        "g711",

        'react',
        'react-dom',
        'react/jsx-runtime',
    ],

    // ── esbuild options ───────────────────────────────────────────────────────────
    esbuildOptions(opts) {
        // Preserve JSDoc comments in output (useful for IDEs)
        opts.legalComments = 'eof'
        // Node built-ins are always external in platform:node
        opts.conditions = ['node', 'import', 'require']
    },

    // ── Watch mode (tsup --watch) ─────────────────────────────────────────────────
    ...(options.watch
        ? {
            onSuccess: 'echo "[@voice-kit/core] rebuilt ✓"',
        }
        : {}),
}))