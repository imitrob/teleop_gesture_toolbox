export class FreshnessTracker {
  constructor(timeoutMs = 500) {
    this.timeoutMs = timeoutMs;
    this.lastSeen = new Map();
  }

  mark(channel, nowMs = performance.now()) {
    this.lastSeen.set(channel, nowMs);
  }

  clear(channel) {
    if (channel === undefined) {
      this.lastSeen.clear();
      return;
    }
    this.lastSeen.delete(channel);
  }

  age(channel, nowMs = performance.now()) {
    const lastSeen = this.lastSeen.get(channel);
    return lastSeen === undefined ? Infinity : Math.max(0, nowMs - lastSeen);
  }

  isFresh(channel, nowMs = performance.now()) {
    return this.age(channel, nowMs) < this.timeoutMs;
  }
}
