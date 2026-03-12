import { configureLogger } from "./logger";

const logger = configureLogger("observableVariable");

type Observer<T> = (value: T) => void | Promise<void>;

export class ObservableVariable<T> {
    private _value: T;
    private _observers: Observer<T>[] = [];

    constructor(value: T) {
        this._value = value;
    }

    /** Register a sync or async observer. */
    addObserver(observer: Observer<T>): void {
        this._observers.push(observer);
    }

    get value(): T {
        return this._value;
    }

    set value(newValue: T) {
        if (this._value !== newValue) {
            this._value = newValue;
            this._notifyObservers(newValue);
        }
    }

    private _notifyObservers(newValue: T): void {
        for (const observer of this._observers) {
            try {
                const result = observer(newValue);
                if (result instanceof Promise) {
                    result.catch((e) =>
                        logger.error(`Async observer error: ${e}`)
                    );
                }
            } catch (e) {
                logger.error(`Observer error: ${e}`);
            }
        }
    }
}