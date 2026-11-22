import type { IncomingMessage, ServerResponse } from "node:http";

export type ModelRequestResult<T> = { status: "completed"; value: T } | { status: "busy" } | { status: "cancelled" };
export type RunModelRequest = <T>(
  request: IncomingMessage,
  response: ServerResponse,
  endpoint: string,
  run: (signal: AbortSignal, markPhase: (phase: string) => void) => Promise<T>,
  timeoutMs?: number,
) => Promise<ModelRequestResult<T>>;
