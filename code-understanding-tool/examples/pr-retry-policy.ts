export type RetryPolicy = {
  baseDelayMs: number;
  maxDelayMs: number;
};

export function retryDelayMs(attempt: number, policy: RetryPolicy): number {
  if (!Number.isInteger(attempt) || attempt < 0) {
    throw new RangeError("attempt must be a non-negative integer");
  }
  if (policy.baseDelayMs < 0 || policy.maxDelayMs < 0) {
    throw new RangeError("delay limits must be non-negative");
  }

  const exponentialDelay = policy.baseDelayMs * 2 ** attempt;
  return Math.min(exponentialDelay, policy.maxDelayMs);
}
