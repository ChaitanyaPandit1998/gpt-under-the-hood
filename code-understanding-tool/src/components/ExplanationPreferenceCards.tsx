import { useState } from "react";
import type { ExplanationDepth, LearnerLevel } from "../domain/explanation-types";
import { ChoiceCards, type CardChoice } from "./ChoiceCards";

const audiences = [
  { value: "beginner", title: "Beginner", description: "Define terms and build up from the basics." },
  { value: "intermediate", title: "Intermediate", description: "Assume the basics and connect the main ideas." },
  { value: "advanced", title: "Advanced", description: "Use precise terms and skip familiar definitions." },
] as const satisfies readonly CardChoice<LearnerLevel>[];
const depths = [
  { value: "overview", title: "Quick overview", description: "Get the key idea and why it matters." },
  { value: "balanced", title: "Step-by-step", description: "Follow the main steps and the reasoning behind them." },
  { value: "detailed", title: "Deep dive", description: "Explore details, edge cases, and trade-offs." },
] as const satisfies readonly CardChoice<ExplanationDepth>[];

export function ExplanationPreferenceCards({ depth, audience, onDepthChange, onAudienceChange }: {
  depth: ExplanationDepth; audience: LearnerLevel; onDepthChange: (value: ExplanationDepth) => void; onAudienceChange: (value: LearnerLevel) => void;
}) {
  const [step, setStep] = useState<"audience" | "depth">("audience");
  return <section className="explanation-preferences" id="explanation-preferences" tabIndex={-1} aria-label="Explanation preferences">
    <div className="choice-steps" role="group" aria-label="Customize explanation">
      <button type="button" aria-pressed={step === "audience"} onClick={() => setStep("audience")}>1. Audience: {audiences.find((choice) => choice.value === audience)?.title}</button>
      <button type="button" aria-pressed={step === "depth"} onClick={() => setStep("depth")}>2. Depth: {depths.find((choice) => choice.value === depth)?.title}</button>
    </div>
    {step === "audience" ? <ChoiceCards key="audience" label="Who is this for?" choices={audiences} value={audience} onChange={onAudienceChange} />
      : <ChoiceCards key="depth" label="How much detail?" choices={depths} value={depth} onChange={onDepthChange} />}
    <p className="choice-help">Optional. Use the current settings or customize each step before generating.</p>
  </section>;
}
