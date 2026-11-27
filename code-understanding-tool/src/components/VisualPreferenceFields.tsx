import { visualInputLimits, visualPreferenceOptions, type VisualPreferences } from "../domain/visual-brief";
import { useId } from "react";
import { visualPresets, type VisualPresetSelection } from "../domain/visual-presets";
import { ToggleCards } from "./ChoiceCards";
import { VisualDesignerStep } from "./VisualDesignerStep";

const labels = { form: "Visual form", audience: "Visual audience", style: "Visual style", target: "Output target" } as const;

export function VisualPreferenceFields({ value, onChange, presets, onPresetsChange, disabled, sourceAvailable }: {
  value: VisualPreferences; onChange: (next: VisualPreferences) => void; disabled: boolean; sourceAvailable: boolean;
  presets: VisualPresetSelection; onPresetsChange: (next: VisualPresetSelection) => void;
}) {
  const dataHelpId = useId();
  const sourceHelpId = useId();
  return <fieldset className="visual-brief-fields" disabled={disabled}>
    <legend className="visually-hidden">Visual preferences</legend>
    <VisualDesignerStep number="01" title="What to teach">
      <ToggleCards label="Teaching approach" choices={visualPresets.additionalContext} values={presets.additionalContext}
        onChange={(additionalContext) => onPresetsChange({ ...presets, additionalContext })} />
    </VisualDesignerStep>
    <VisualDesignerStep number="02" title="Look and feel">
      <ToggleCards label="Art direction presets" choices={visualPresets.direction} values={presets.direction}
        onChange={(direction) => onPresetsChange({ ...presets, direction })} />
    </VisualDesignerStep>
    <details className="visual-builder-options">
    <summary>Customize visual <small>Optional</small></summary>
    <div className="visual-builder-step">
    <label className="visual-brief-field"><span>What should the visual explain? <small>Optional</small></span>
      <input value={value.question} maxLength={visualInputLimits.question} placeholder="Leave blank to use this explanation's main idea" onChange={(event) => onChange({ ...value, question: event.target.value })} />
    </label>
    <div className="visual-brief-options">
      {(Object.keys(labels) as (keyof typeof labels)[]).map((key) => <label className="visual-brief-field" key={key}>{labels[key]}
        <select aria-label={labels[key]} value={value[key]} onChange={(event) => onChange({ ...value, [key]: event.target.value })}>
          {visualPreferenceOptions[key].map(([id, title]) => <option key={id} value={id}>{title}</option>)}
        </select>
      </label>)}
    </div>
    <label className="visual-brief-field"><span>Art direction or constraints <small>Optional</small></span>
      <textarea rows={2} value={value.direction} maxLength={visualInputLimits.direction} placeholder="For example: use a warm palette and keep labels readable on a printed page" onChange={(event) => onChange({ ...value, direction: event.target.value })} />
    </label>
    <label className="visual-brief-field"><span>Additional context or data <small>Optional</small></span>
      <textarea aria-describedby={dataHelpId} rows={3} value={value.additionalContext} maxLength={visualInputLimits.additionalContext} placeholder="Add facts, example values, labels, or other details the visual should use" onChange={(event) => onChange({ ...value, additionalContext: event.target.value })} />
    </label>
    <p className="visual-context-help" id={dataHelpId}>Paste the exact values or labels the image should show, even if they were in your original input. You can share just these details without attaching the whole source.</p>
    </div>
    </details>
    {sourceAvailable && <div>
      <label className="visual-brief-consent">
        <input aria-describedby={sourceHelpId} type="checkbox" checked={value.includeSource} onChange={(event) => onChange({ ...value, includeSource: event.target.checked })} />
        Include original source for line references
      </label>
      <p className="visual-context-help" id={sourceHelpId}>This attaches the original input with numbered lines. Leave it unchecked to use the explanation and the design instructions only.</p>
    </div>}
  </fieldset>;
}
