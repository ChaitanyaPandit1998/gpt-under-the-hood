export type DiagramExportLayout = "detailed" | "automatic";

export const diagramExportLimits = {
  detailed: { nodes: 8, edges: 12 },
  automatic: { nodes: 16, edges: 24 },
} as const;
