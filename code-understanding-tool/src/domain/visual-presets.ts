export const visualPresets = {
  direction: [
    { value: "Warm palette", title: "Warm palette", description: "Use warm accents with readable contrast.",
      instruction: "Use a warm accent palette with readable contrast." },
    { value: "Editorial illustration", title: "Editorial illustration", description: "Build around a focal image instead of defaulting to boxes.",
      instruction: "Build around a focal image rather than generic boxes where the visual form allows." },
    { value: "Print-friendly", title: "Print-friendly", description: "Use clear labels and make color optional for understanding.",
      instruction: "Use clear labels, strong contrast, and cues that work without color." },
  ],
  additionalContext: [
    { value: "Step by step", title: "Step by step", description: "Guide readers with short numbered callouts.",
      instruction: "When the material has an ordered progression, guide readers with short numbered callouts." },
    { value: "Worked example", title: "Worked example", description: "Use a supplied example, or clearly label an illustrative one.",
      instruction: "Prefer a supplied worked example. Clearly label any illustrative example as hypothetical." },
    { value: "Key takeaway", title: "Key takeaway", description: "Emphasize one idea and move supporting detail to the caption.",
      instruction: "Emphasize one central takeaway and put supporting detail in the caption where appropriate." },
  ],
} as const;

export type VisualPresetGroup = keyof typeof visualPresets;
export type VisualPresetSelection = { [K in VisualPresetGroup]: readonly typeof visualPresets[K][number]["value"][] };

export function combineVisualPresets(group: VisualPresetGroup, selected: readonly string[], custom: string) {
  return [...visualPresets[group].filter((preset) => selected.includes(preset.value)).map((preset) => preset.instruction), custom.trim()]
    .filter(Boolean).join("\n");
}
