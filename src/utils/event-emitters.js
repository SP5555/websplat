'use strict';

class EventEmitter {
    constructor() {
        this.listeners = {};
    }

    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = new Set();
        }
        this.listeners[event].add(callback);
        return () => this.off(event, callback);
    }

    off(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event].delete(callback);
            if (this.listeners[event].size === 0) {
                delete this.listeners[event];
            }
        }
    }

    once(event, callback) {
        const onceAndDel = (...args) => {
            callback(...args);
            this.off(event, onceAndDel);
        };
        this.on(event, onceAndDel);
    }

    emit(event, ...args) {
        // console.log(`Event:${event}`);
        if (this.listeners[event]) {
            for (const callback of this.listeners[event]) {
                callback(...args);
            }
        }
    }

    clear() {
        this.listeners = {};
    }
}

// use only for cross module communications
export const eventBus = new EventEmitter();