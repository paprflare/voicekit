import { configureLogger } from "../helper/logger";
import { BaseAgent } from "./base";

const logger = configureLogger("webhookAgent");

export class WebhookAgent extends BaseAgent {
    private webhookUrl: string;
    private payload: Record<string, unknown>;

    constructor(webhookUrl: string, payload: Record<string, unknown> = {}) {
        super();
        this.webhookUrl = webhookUrl;
        this.payload = payload;
    }

    private async sendPayload(
        payload: Record<string, unknown> | null
    ): Promise<true | null> {
        try {
            logger.info(`Sending a webhook post request ${JSON.stringify(payload)}`);
            if (payload === null) {
                logger.info("Payload was null");
                return null;
            }
            const res = await fetch(this.webhookUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (res.ok) return true;
            logger.error(`Error: ${res.status} - ${await res.text()}`);
            return null;
        } catch (e) {
            logger.error(`Something went wrong with webhook ${this.webhookUrl}, ${JSON.stringify(payload)}, ${e}`);
            return null;
        }
    }

    async execute(payload: Record<string, unknown> | null): Promise<true | null> {
        if (!this.webhookUrl) return null;
        const response = await this.sendPayload(payload);
        logger.info(`Response ${response}`);
        return response;
    }
}