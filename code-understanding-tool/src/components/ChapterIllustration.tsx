import { useEffect, useId, useRef, useState } from "react";
import { artworkThemes, chapterArtworkPrompt, type ArtworkTheme, type ChapterArtwork } from "../domain/chapter-illustrations";

export function ChapterIllustration({ cached, onChange }: { cached?: ChapterArtwork; onChange: (artwork: ChapterArtwork | null) => void }) {
  const [theme, setTheme] = useState<ArtworkTheme>(cached?.theme ?? "review");
  const [approved, setApproved] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [visible, setVisible] = useState(true);
  const controller = useRef<AbortController | null>(null);
  const requestId = useRef(0);
  const id = useId();
  useEffect(() => () => { requestId.current += 1; controller.current?.abort(); }, []);

  function cancel() {
    requestId.current += 1; controller.current?.abort(); controller.current = null; setLoading(false); setApproved(false);
  }

  async function generate() {
    if (!approved || loading) return;
    const currentId = ++requestId.current;
    const current = new AbortController();
    controller.current = current;
    setLoading(true); setError("");
    try {
      const response = await fetch("/api/chapter-cover", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ theme, approved: true }), signal: current.signal });
      const payload = await response.json() as ChapterArtwork & { error?: string };
      if (currentId !== requestId.current) return;
      if (!response.ok) throw new Error(payload.error || "Could not generate chapter artwork.");
      if (payload.theme !== theme || typeof payload.dataUrl !== "string" || !/^data:image\/(jpeg|png);base64,[a-z0-9+/]+=*$/i.test(payload.dataUrl)) throw new Error("The artwork response was invalid.");
      onChange(payload); setVisible(true);
    } catch (caught) {
      if (currentId === requestId.current && !current.signal.aborted) setError(caught instanceof Error ? caught.message : "Artwork generation failed.");
    } finally {
      if (currentId === requestId.current) { setLoading(false); setApproved(false); controller.current = null; }
    }
  }

  return <section className="chapter-artwork" aria-label="Optional chapter cover">
    <details><summary>Optional chapter cover</summary>
      <p>Decorative artwork can provide a visual metaphor, but cannot establish code behaviour. Use the source-backed walkthrough, control-flow maps, and citations for branches, ordering, and outcomes.</p>
      <label htmlFor={`${id}-theme`}>Generic artwork theme</label>
      <select id={`${id}-theme`} disabled={loading} value={theme} onChange={(event) => { setTheme(event.target.value as ArtworkTheme); setApproved(false); setError(""); }}>
        {Object.entries(artworkThemes).map(([value, item]) => <option key={value} value={value}>{item.label}</option>)}
      </select>
      <details className="artwork-prompt"><summary>Exact prompt sent to Cloudflare</summary><pre>{chapterArtworkPrompt(theme)}</pre></details>
      <p>Only this generic theme is sent. No chapter title, summary, PR link, or source code is included. Generation may use free allocation or incur charges. Cancelling may not prevent provider billing once work starts.</p>
      <label className="artwork-consent"><input type="checkbox" checked={approved} disabled={loading} onChange={(event) => setApproved(event.target.checked)} />I approve sending this generic artwork prompt to Cloudflare and any applicable generation charges.</label>
      <div className="chapter-toolbar"><button type="button" disabled={!approved || loading} onClick={() => void generate()}>{loading ? "Generating cover…" : cached ? "Regenerate cover" : "Generate cover"}</button>{loading && <button type="button" onClick={cancel}>Cancel cover generation</button>}{cached && <button type="button" onClick={() => { cancel(); onChange(null); }}>Remove cover</button>}</div>
      {loading && <p role="status">Generating conceptual artwork. The code explanation is unchanged.</p>}
      {error && <p role="alert">{error}</p>}
    </details>
    {cached && <><button className="artwork-visibility" type="button" onClick={() => setVisible(!visible)}>{visible ? "Hide cover" : "Show cover"}</button>{visible && <figure>
      <img src={cached.dataUrl} alt={`Decorative isometric illustration: ${artworkThemes[cached.theme].label}. Not a code-behaviour diagram.`} />
      <figcaption><strong>Conceptual illustration</strong> · {artworkThemes[cached.theme].label}. AI-generated decorative artwork, not verified code behaviour. Review before reuse.</figcaption>
    </figure>}</>}
  </section>;
}
