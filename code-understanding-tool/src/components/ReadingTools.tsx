import { useState, type ReactNode } from "react";
import type { Explanation } from "../domain/explanation-types";
import { type ExportContext, DocumentExport } from "./DocumentExport";
import { SpeechReader } from "./SpeechReader";
import { type StudyPdfImages, StudyPdfExport } from "./StudyPdfExport";

export function ReadingTools({ label, scope, explanation, source, context, refine, visual, collectImages, onlyChapterId, chapterSpeech, children }: {
  label: string; scope: string; explanation: Explanation; source: string; context: ExportContext;
  refine: ReactNode; visual: ReactNode; collectImages: () => StudyPdfImages;
  onlyChapterId?: string; chapterSpeech?: Explanation; children?: ReactNode;
}) {
  const [listeningOpen, setListeningOpen] = useState(false);
  return <div className="reading-tools scoped-reading-tools" role="group" aria-label={label}>
    <div className="section-caption">{scope}</div>
    {refine}
    {visual}
    <details className="speech-tools" onToggle={(event) => setListeningOpen(event.currentTarget.open)}>
      <summary>Read aloud</summary>
      {listeningOpen && <SpeechReader explanation={chapterSpeech ?? explanation} chapterOnly={!!chapterSpeech} />}
    </details>
    {!onlyChapterId && <DocumentExport explanation={explanation} source={source} context={context} />}
    <StudyPdfExport explanation={explanation} source={source} context={context} collectImages={collectImages} onlyChapterId={onlyChapterId} />
    {children}
  </div>;
}
