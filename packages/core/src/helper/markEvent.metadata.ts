import { configureLogger } from "./logger";

const logger = configureLogger("markEventMetaData");

export class MarkEventMetaData {
    private markEventMetaData: Record<string, Record<string, unknown>> = {};
    private previousMarkEventMetaData: Record<string, Record<string, unknown>> = {};
    private counter = 0;

    updateData(markId: string, value: Record<string, unknown>): void {
        value["counter"] = this.counter++;
        this.markEventMetaData[markId] = value;
    }

    fetchData(markId: string): Record<string, unknown> {
        const data = this.markEventMetaData[markId] ?? {};
        delete this.markEventMetaData[markId];
        return data;
    }

    clearData(): void {
        logger.info("Clearing mark meta data dict");
        this.counter = 0;
        this.previousMarkEventMetaData = structuredClone(this.markEventMetaData);
        this.markEventMetaData = {};
    }

    fetchClearedMarkEventData(): Record<string, Record<string, unknown>> {
        return this.previousMarkEventMetaData;
    }

    toString(): string {
        return JSON.stringify(this.markEventMetaData);
    }
    getMarkEventMetaData(): Record<string, unknown> {
        return this.markEventMetaData;
    }
}