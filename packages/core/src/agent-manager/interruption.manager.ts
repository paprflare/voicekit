import { configureLogger } from "../helper/logger";

const logger = configureLogger("interruptionManager");

export class InterruptionManager {
    // User speaking state
    calleeSpeaking: boolean = false;
    private calleeSpeakingStartTime: number = -1;

    // Sequence management (-1 reserved for background audio)
    private currSequenceId: number = 0;
    sequenceIds: Set<number> = new Set([-1]);

    // Turn tracking
    private turnId: number = 0;

    // Timing state
    letRemainingAudioPassThrough: boolean = false;
    private timeSinceFirstInterimResult: number = -1;
    private requiredDelayBeforeSpeaking: number = 0;
    private incrementalDelay: number;
    private utteranceEndTime: number = -1;

    // Configuration
    private numberOfWordsForInterruption: number;
    private accidentalInterruptionPhrases: Set<string>;
    private minimumWaitDuration: number;

    constructor(opts: {
        numberOfWordsForInterruption?: number;
        accidentalInterruptionPhrases?: string[];
        incrementalDelay?: number;
        minimumWaitDuration?: number;
    } = {}) {
        this.numberOfWordsForInterruption = opts.numberOfWordsForInterruption ?? 3;
        this.accidentalInterruptionPhrases = new Set(opts.accidentalInterruptionPhrases ?? []);
        this.incrementalDelay = opts.incrementalDelay ?? 900;
        this.minimumWaitDuration = opts.minimumWaitDuration ?? 0;

        logger.info(
            `InterruptionManager initialized: ` +
            `words_for_interruption=${this.numberOfWordsForInterruption}, ` +
            `incremental_delay=${this.incrementalDelay}ms`
        );
    }

    // ------------------------------------------------------------------
    // Audio send decision
    // ------------------------------------------------------------------

    getAudioSendStatus(sequenceId: number, historyLength: number = 0): "SEND" | "BLOCK" | "WAIT" {
        // Check 1: Invalid sequence — discard
        if (!this.sequenceIds.has(sequenceId)) return "BLOCK";

        // Check 2: User is speaking — hold audio
        if (this.calleeSpeaking) {
            logger.info("Audio status=WAIT - user is speaking");
            return "WAIT";
        }

        // Check 3: Grace period (only after first 2 turns to avoid latency on welcome)
        if (historyLength > 2) {
            const timeSinceUtteranceEnd = this.getTimeSinceUtteranceEnd();
            if (timeSinceUtteranceEnd !== -1 && timeSinceUtteranceEnd < this.incrementalDelay) {
                logger.info(
                    `Audio status=WAIT - grace period: ${timeSinceUtteranceEnd.toFixed(0)}ms / ${this.incrementalDelay}ms`
                );
                return "WAIT";
            }
        }

        return "SEND";
    }

    // ------------------------------------------------------------------
    // Interruption logic
    // ------------------------------------------------------------------

    shouldTriggerInterruption(
        wordCount: number,
        transcript: string,
        isAudioPlaying: boolean,
        welcomePlayed: boolean
    ): boolean {
        if (!isAudioPlaying || !welcomePlayed) return false;
        if (this.numberOfWordsForInterruption === 0) return false;

        const transcriptStripped = transcript.trim();
        return (
            wordCount > this.numberOfWordsForInterruption ||
            this.accidentalInterruptionPhrases.has(transcriptStripped)
        );
    }

    isFalseInterruption(
        wordCount: number,
        transcript: string,
        isAudioPlaying: boolean,
        welcomePlayed: boolean
    ): boolean {
        if (!isAudioPlaying || !welcomePlayed) return false;

        const transcriptStripped = transcript.trim();
        return (
            wordCount <= this.numberOfWordsForInterruption &&
            !this.accidentalInterruptionPhrases.has(transcriptStripped)
        );
    }

    // ------------------------------------------------------------------
    // Speech lifecycle events
    // ------------------------------------------------------------------

    onUserSpeechStarted(): void {
        if (!this.calleeSpeaking) {
            this.calleeSpeaking = true;
            this.calleeSpeakingStartTime = Date.now() / 1000;
            logger.info("User started speaking");
        }
    }

    onInterimTranscriptReceived(): void {
        this.letRemainingAudioPassThrough = false;

        if (this.timeSinceFirstInterimResult === -1) {
            this.timeSinceFirstInterimResult = Date.now();
            logger.info(`First interim at ${this.timeSinceFirstInterimResult}`);
        }
    }

    onUserSpeechEnded(updateUtteranceTime: boolean = true): void {
        this.calleeSpeaking = false;
        this.letRemainingAudioPassThrough = true;
        this.timeSinceFirstInterimResult = -1;
        if (updateUtteranceTime) {
            this.utteranceEndTime = Date.now();
        }
        logger.info("User speech ended");
    }

    onInterruptionTriggered(): void {
        this.turnId += 1;
        this.invalidatePendingResponses();
        logger.info(`Interruption triggered, turn_id=${this.turnId}`);
    }

    // ------------------------------------------------------------------
    // Sequence management
    // ------------------------------------------------------------------

    invalidatePendingResponses(): void {
        this.sequenceIds = new Set([-1]);
        logger.info("Pending responses invalidated");
    }

    revalidateSequenceId(sequenceId: number): void {
        this.sequenceIds.add(sequenceId);
        logger.info(`Re-validated sequence_id=${sequenceId}`);
    }

    getNextSequenceId(): number {
        this.currSequenceId += 1;
        this.sequenceIds.add(this.currSequenceId);
        return this.currSequenceId;
    }

    isValidSequence(sequenceId: number): boolean {
        return this.sequenceIds.has(sequenceId);
    }

    hasPendingResponses(): boolean {
        return this.sequenceIds.size > 1;
    }

    // ------------------------------------------------------------------
    // Delay logic
    // ------------------------------------------------------------------

    shouldDelayOutput(welcomeMessagePlayed: boolean): [boolean, number] {
        if (!welcomeMessagePlayed) return [false, 0];
        return this.checkDelay();
    }

    private checkDelay(): [boolean, number] {
        if (this.timeSinceFirstInterimResult === -1) return [false, 0];

        const elapsed = Date.now() - this.timeSinceFirstInterimResult;
        if (elapsed < this.requiredDelayBeforeSpeaking) return [true, 0.1];

        return [false, 0];
    }

    updateRequiredDelay(historyLength: number): void {
        this.requiredDelayBeforeSpeaking =
            historyLength > 2 ? this.incrementalDelay : 0;
    }

    resetDelayForSpeechFinal(historyLength: number): void {
        this.timeSinceFirstInterimResult = -1;
        this.requiredDelayBeforeSpeaking =
            historyLength > 2
                ? Math.max(this.minimumWaitDuration - this.incrementalDelay, 0)
                : 0;
    }

    // ------------------------------------------------------------------
    // Timing getters
    // ------------------------------------------------------------------

    getTimeSinceUtteranceEnd(): number {
        if (this.utteranceEndTime === -1) return -1;
        return Date.now() - this.utteranceEndTime;
    }

    resetUtteranceEndTime(): void {
        this.utteranceEndTime = -1;
        logger.info("Utterance end time reset");
    }

    getUserSpeakingDuration(): number {
        if (!this.calleeSpeaking || this.calleeSpeakingStartTime < 0) return 0;
        return Date.now() / 1000 - this.calleeSpeakingStartTime;
    }

    setFirstInterimForImmediateResponse(): void {
        this.timeSinceFirstInterimResult = Date.now() - 1000;
    }

    // ------------------------------------------------------------------
    // Simple getters
    // ------------------------------------------------------------------

    isUserSpeaking(): boolean { return this.calleeSpeaking; }
    getTurnId(): number { return this.turnId; }
}