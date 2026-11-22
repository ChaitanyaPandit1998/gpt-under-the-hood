import { useId } from "react";

export type CardChoice<T extends string> = { value: T; title: string; description: string };

type GroupProps<T extends string> = { label: string; choices: readonly CardChoice<T>[]; disabled?: boolean };

export function ChoiceCards<T extends string>({ value, onChange, ...group }: GroupProps<T> & {
  value: T; onChange: (value: T) => void;
}) {
  return <CardGroup {...group} type="radio" isSelected={(choice) => choice === value} onToggle={onChange} />;
}

export function ToggleCards<T extends string>({ values, onChange, ...group }: GroupProps<T> & {
  values: readonly T[]; onChange: (values: T[]) => void;
}) {
  return <CardGroup {...group} type="checkbox" isSelected={(choice) => values.includes(choice)}
    onToggle={(choice) => onChange(values.includes(choice) ? values.filter((value) => value !== choice) : [...values, choice])} />;
}

function CardGroup<T extends string>({ label, choices, type, disabled, isSelected, onToggle }: GroupProps<T> & {
  type: "radio" | "checkbox"; isSelected: (value: T) => boolean; onToggle: (value: T) => void;
}) {
  const group = useId();
  return <fieldset className="choice-cards" disabled={disabled}>
    <legend>{label}</legend>
    <div className="choice-cards-grid">
      {choices.map((choice, index) => <label className="choice-card" key={choice.value}>
        <input type={type} name={group} checked={isSelected(choice.value)} value={choice.value}
          aria-labelledby={`${group}-title-${index}`} aria-describedby={`${group}-description-${index}`}
          onChange={() => onToggle(choice.value)} />
        <strong id={`${group}-title-${index}`}>{choice.title}</strong>
        <span id={`${group}-description-${index}`}>{choice.description}</span>
      </label>)}
    </div>
  </fieldset>;
}
