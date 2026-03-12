export abstract class BaseCache {
    abstract set(...args: unknown[]): unknown;
    abstract get(...args: unknown[]): unknown;
}