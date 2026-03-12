import { LRUCache } from "lru-cache";
import { BaseCache } from "./base";
import { configureLogger } from "../../helper/logger";

const logger = configureLogger("inmemoryScalarCache");

interface CacheEntry<T> {
    value: T;
    setAt: number;
}

export class InmemoryScalarCache<T = unknown> extends BaseCache {
    private cache: LRUCache<string, CacheEntry<T>>;
    private ttl: number;

    constructor(ttl = -1, maxSize = 500) {
        super();
        this.ttl = ttl;
        this.cache = new LRUCache<string, CacheEntry<T>>({
            max: maxSize,
            // Let LRU handle TTL natively when ttl > 0
            ...(ttl > 0 ? { ttl: ttl * 1000 } : {}),
        });
    }

    get(key: string): T | null {
        const entry = this.cache.get(key);

        if (!entry) {
            logger.info(`Cache miss for key ${key}`);
            return null;
        }

        // Manual TTL check for ttl == -1 (permanent) or custom logic
        if (this.ttl === -1) {
            return entry.value;
        }

        const ageMs = Date.now() - entry.setAt;
        if (ageMs < this.ttl * 1000) {
            return entry.value;
        }

        logger.info(`Cache miss for key ${key}`);
        return null;
    }

    set(key: string, value: T): void {
        this.cache.set(key, { value, setAt: Date.now() });
    }

    flushCache(onlyEphemeral = true): void {
        if (onlyEphemeral) {
            this.cache.clear();
        } else {
            // Clear only TTL-tracked entries (non-permanent)
            for (const [key, entry] of this.cache.entries()) {
                if (this.ttl !== -1) {
                    this.cache.delete(key);
                }
            }
        }
    }
}