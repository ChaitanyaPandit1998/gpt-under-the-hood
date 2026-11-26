import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Explanation } from "../domain/explanation-types";
import { type ExportContext, DocumentExport } from "./DocumentExport";
import { SpeechReader } from "./SpeechReader";
import { type StudyPdfImages, StudyPdfExport } from "./StudyPdfExport";

export function ReadingTools({ label, scope, explanation, source, context, refine, visual, collectImages, onlyChapterId, chapterSpeech, saveStudy, targetId }: {
  label: string; scope: string; explanation: Explanation; source: string; context: ExportContext;
  refine: ReactNode; visual: ReactNode; collectImages: () => StudyPdfImages;
  onlyChapterId?: string; chapterSpeech?: Explanation; saveStudy?: { action: ReactNode; note: ReactNode }; targetId?: string;
}) {
  const [listeningOpen, setListeningOpen] = useState(false);
  const toolbar = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = toolbar.current;
    if (!container) return;
    function revealTool(event: Event) {
      const opened = event.target;
      if (!(opened instanceof HTMLDetailsElement) || !opened.open || opened.parentElement !== container) return;
      for (const sibling of container!.children) {
        if (sibling !== opened && sibling instanceof HTMLDetailsElement && sibling.open
          && !sibling.querySelector('[data-tool-active="true"], [data-tool-decision="true"]') && sibling.dataset.toolActive !== "true") {
          sibling.open = false;
        }
      }
    }
    // Native toggle does not bubble; capture only direct tool disclosures, not nested options.
    container.addEventListener("toggle", revealTool, true);
    return () => container.removeEventListener("toggle", revealTool, true);
  }, []);
  const pdf = <StudyPdfExport explanation={explanation} source={source} context={context} collectImages={collectImages} onlyChapterId={onlyChapterId} />;
  const documentExport = !onlyChapterId && <DocumentExport explanation={explanation} source={source} context={context} />;
  return <div ref={toolbar} id={targetId} tabIndex={targetId ? -1 : undefined} className="reading-tools scoped-reading-tools" role="group" aria-label={label}>
    <div className="section-caption">{scope}</div>
    {refine}
    {visual}
    <details className="speech-tools" onToggle={(event) => { if (event.currentTarget.open) setListeningOpen(true); }}>
      <summary>Read aloud <small className="tool-active-indicator">Playback active</small></summary>
      {listeningOpen && <SpeechReader explanation={chapterSpeech ?? explanation} chapterOnly={!!chapterSpeech} />}
    </details>
    {saveStudy ? <details className="study-actions">
      <summary><span>Save &amp; export</span> <small className="tool-active-indicator">Exporting</small></summary>
      <div className="study-actions-content">
        <div className="study-actions-buttons">{saveStudy.action}{pdf}</div>
        {saveStudy.note}
        {documentExport}
      </div>
    </details> : <>{documentExport}{pdf}</>}
  </div>;
}
