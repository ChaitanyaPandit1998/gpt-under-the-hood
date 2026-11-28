import { useState, type FormEvent } from "react";
import { parseStoryCast, storyCastNameLimit, storyHeadChoices, type StoryCast } from "../domain/story-cast";

export type StoryCastControl = { value?: StoryCast; onChange: (cast?: StoryCast) => void };

export function StoryCastSettings({ value, onChange }: StoryCastControl) {
  const [draft, setDraft] = useState<StoryCast>(() => value ?? {});
  const [error, setError] = useState("");

  function apply(event: FormEvent) {
    event.preventDefault();
    const roles = storyHeadChoices.flatMap(({ id }) => {
      const name = draft[id]?.trim();
      return name ? [[id, name]] : [];
    });
    try {
      onChange(parseStoryCast(roles.length ? Object.fromEntries(roles) : undefined));
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Invalid cast.");
    }
  }

  return <details className="visual-builder-options story-cast-settings">
    <summary>{value ? "Story cast: locked" : "Lock character appearances"}</summary>
    <p>Assign either face to a fictional role used in your Story. This applies to new illustrations in this study, not existing images or factual diagrams.</p>
    <form onSubmit={apply}>
      {storyHeadChoices.map(({ id, label }) => <label key={id}>
        {label}
        <input value={draft[id] ?? ""} placeholder="Fictional name or role" maxLength={storyCastNameLimit}
          onChange={(event) => setDraft({ ...draft, [id]: event.target.value })} />
      </label>)}
      <div className="chapter-toolbar">
        <button type="submit">Apply cast</button>
        {value && <button type="button" onClick={() => { setDraft({}); setError(""); onChange(undefined); }}>Unlock cast</button>}
        <span role="status">{value ? "Cast locked for future illustrations." : "Automatic casting"}</span>
      </div>
      {error && <p role="alert">{error}</p>}
    </form>
  </details>;
}
