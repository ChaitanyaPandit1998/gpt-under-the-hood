import { useId, useLayoutEffect, useRef } from "react";
import type { SourceRange } from "../domain/source-ranges";

const sheetMedia = "(max-width: 900px)";

function revealCitation(excerpt: HTMLDivElement | null) {
  const first = excerpt?.querySelector("mark");
  if (!excerpt || !first) return;
  const offset = first.getBoundingClientRect().top - excerpt.getBoundingClientRect().top + excerpt.scrollTop;
  excerpt.scrollTo({ top: Math.max(0, offset - excerpt.clientHeight / 4), left: 0, behavior: "instant" });
}

export function SourceEvidence({ source, range, isPatch, onDismiss }: {
  source: string; range: SourceRange; isPatch: boolean; onDismiss: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const excerpt = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const lines = source.split(/\r?\n/);
  const start = Math.max(1, range.start_line - 4);
  const end = Math.min(lines.length, range.end_line + 4);
  const label = range.start_line === range.end_line ? `Line ${range.start_line}` : `Lines ${range.start_line}–${range.end_line}`;

  useLayoutEffect(() => {
    const panel = dialog.current;
    if (!panel) return;
    const media = window.matchMedia(sheetMedia);
    function open() {
      if (!panel) return;
      if (panel.open) panel.close();
      if (media.matches) panel.showModal();
      else panel.show();
      revealCitation(excerpt.current);
    }
    open();
    media.addEventListener("change", open);
    return () => media.removeEventListener("change", open);
  }, []);

  useLayoutEffect(() => {
    revealCitation(excerpt.current);
    dialog.current?.querySelector("button")?.focus({ preventScroll: true });
  }, [range]);

  return <dialog className="source-evidence" ref={dialog} aria-labelledby={titleId}
    onCancel={(event) => { event.preventDefault(); onDismiss(); }}
    onKeyDown={(event) => {
      if (event.key === "Escape") { event.preventDefault(); onDismiss(); }
    }}>
    <header>
      <div><div className="section-caption">CITED EVIDENCE</div><h3 id={titleId}>{isPatch ? "PR chapter source" : "Source reference"}</h3></div>
      <button type="button" onClick={onDismiss}>Close source</button>
    </header>
    <p><strong>{label}</strong>{isPatch ? " · Numbers refer to the imported chapter, not the original file." : " · Highlighted in the original input."}</p>
    <div className="source-evidence-lines" ref={excerpt} role="region" aria-label="Source excerpt" tabIndex={0}>
      {lines.slice(start - 1, end).map((line, index) => {
        const number = start + index;
        const Tag = number >= range.start_line && number <= range.end_line ? "mark" : "div";
        return <Tag className="source-evidence-line" key={number}><span>{number}</span><code>{line}</code></Tag>;
      })}
    </div>
  </dialog>;
}
