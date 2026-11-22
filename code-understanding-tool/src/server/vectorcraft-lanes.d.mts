import type { Explanation } from "../domain/explanation-types.ts";

export function diagramPlan(visual: Explanation["visuals"][number], sourceLabel: string, lineLabel?: string): {
  commands: { command: string; params: Record<string, unknown> }[];
  width: number;
  height: number;
  mode: "lanes" | "connection-index";
};
