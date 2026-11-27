import { useRef, useState } from "react";
import type { SavedStudy } from "../domain/saved-study";
import { exportPortableStudy, importPortableStudy } from "../domain/portable-study";

export function PortableStudyTools({ study, disabled, onImport }: {
  study?: SavedStudy; disabled: boolean; onImport: (study: SavedStudy) => Promise<void>;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  async function transfer(file?: File) {
    if (busy || disabled) return;
    if (!file && (!study || !window.confirm("This file includes saved source text, explanations, stories, and images. Share it only with people you trust. Export?"))) return;
    setBusy(true); setMessage("");
    try {
      if (file) {
        const imported = await importPortableStudy(file);
        await onImport(imported);
      } else if (study) {
        const blob = await exportPortableStudy(study);
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url; link.download = "understand-study.json"; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        setMessage("Saved study exported. Unsaved changes are not included.");
      }
    } catch (error) { setMessage(error instanceof Error ? error.message : "Could not transfer the study."); }
    finally { setBusy(false); if (input.current) input.current.value = ""; }
  }
  return <div className="saved-study-tools" aria-label="Portable study files">
    <div className="chapter-toolbar">
      <button type="button" disabled={disabled || busy || !study || study.mode === "pull_request"} onClick={() => void transfer()}>Export saved study</button>
      <button type="button" disabled={disabled || busy} onClick={() => input.current?.click()}>Import study file</button>
      <input hidden ref={input} type="file" accept=".json,application/json" aria-label="Study file" onChange={event => { const file = event.target.files?.[0]; if (file) void transfer(file); }} />
    </div>
    <p>Transfer a saved code or document study with its stories, images and review context/history. Files include source text and may quote it in review materials. Imports stay local; use Resume saved study to open them. Limit: 40 MB. PR studies and decorative covers are not supported yet.</p>
    {message && <p role="status">{message}</p>}
  </div>;
}
