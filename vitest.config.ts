import { defineConfig } from 'vitest/config'

/**
 * Root Vitest configuration for @voice-kit monorepo.
 *
 * Uses the `projects` array (vitest ≥ 1.3 / stable in v2+).
 * Replaces the deprecated vitest.workspace.ts approach.
 *
 * Run all:         vitest run
 * Watch:           vitest
 * Single package:  vitest run --project core
 * Coverage:        vitest run --coverage
 */
export default defineConfig({
    test: {
        projects: [
            'packages/core',
            'packages/agents',
            'packages/stream',
            'packages/telephony',
            'packages/react',
        ],

        // Root-level coverage — aggregates all packages
        coverage: {
            provider: 'v8',
            include: ['packages/*/src/**/*.ts'],
            exclude: [
                'packages/*/src/**/index.ts',
                'packages/*/src/**/*.d.ts',
                '**/node_modules/**',
                '**/dist/**',
            ],
            reporter: ['text', 'html', 'lcov'],
            reportsDirectory: "coverage",
        },
    },
})