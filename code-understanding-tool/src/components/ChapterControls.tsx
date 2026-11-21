import { adjustChapterDepth, type ChapterSettings } from "../domain/chapter-settings";

function ControlIcon({ kind }: { kind: "less" | "more" | "settings" }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    {kind === "settings" ? <><path d="M4 7h16M4 17h16" /><circle cx="9" cy="7" r="3" fill="var(--card-bg, white)" /><circle cx="15" cy="17" r="3" fill="var(--card-bg, white)" /></> : <><path d="M5 12h14" />{kind === "more" && <path d="M12 5v14" />}</>}
  </svg>;
}

export function ChapterControls({ settings, onChange, disabled = false }: { settings: ChapterSettings; onChange: (settings: ChapterSettings) => void; disabled?: boolean }) {
  return <div className="chapter-controls">
    <div className="chapter-quick-controls" role="group" aria-label="Chapter explanation controls">
      <button type="button" disabled={disabled || settings.depth === "overview"} onClick={() => onChange(adjustChapterDepth(settings, -1))}><ControlIcon kind="less" />Simplify</button>
      <div className="depth-meter" role="img" aria-label={`Explanation depth: ${settings.depth}`}>{["overview", "balanced", "detailed"].map((depth, index) => <span key={depth} data-active={index <= ["overview", "balanced", "detailed"].indexOf(settings.depth)} />)}</div>
      <button type="button" disabled={disabled || settings.depth === "detailed"} onClick={() => onChange(adjustChapterDepth(settings, 1))}><ControlIcon kind="more" />Go deeper</button>
    </div>
    <details><summary><ControlIcon kind="settings" />Customize explanation</summary>
      <fieldset disabled={disabled} className="chapter-setting-fields"><legend className="visually-hidden">Chapter preferences</legend>
        <label>Depth<select aria-label="Chapter depth" value={settings.depth} onChange={(event) => onChange({ ...settings, depth: event.target.value as ChapterSettings["depth"] })}><option value="overview">Overview</option><option value="balanced">Balanced</option><option value="detailed">Detailed</option></select></label>
        <label>Explain for<select aria-label="Chapter learner level" value={settings.learnerLevel} onChange={(event) => onChange({ ...settings, learnerLevel: event.target.value as ChapterSettings["learnerLevel"] })}><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option></select></label>
        <label>Tone<select aria-label="Chapter tone" value={settings.tone} onChange={(event) => onChange({ ...settings, tone: event.target.value as ChapterSettings["tone"] })}><option value="plain">Plain language</option><option value="technical">Technical</option></select></label>
        <label>Focus<select aria-label="Chapter focus" value={settings.focus} onChange={(event) => onChange({ ...settings, focus: event.target.value as ChapterSettings["focus"] })}><option value="auto">Balanced</option><option value="concepts">Concepts</option><option value="behaviour">Step-by-step behaviour</option><option value="example">Worked example</option></select></label>
      </fieldset>
    </details>
    <small>Only this chapter changes. Choose settings, then generate; cached variants need no new request. More detail does not mean more certainty.</small>
  </div>;
}
