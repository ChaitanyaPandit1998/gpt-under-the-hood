import { createContext, useContext, useId, useRef, useState, type ReactNode } from "react";
import type { Explanation } from "../domain/explanation-types";
import { explanationChapters, prefersChapters } from "../domain/explanation-chapters";

const ChapterContext = createContext<string | null>(null);

export function ChapterPage({ id, children }: { id: string; children: ReactNode }) {
  const activeChapter = useContext(ChapterContext);
  return <div className="chapter-page" hidden={activeChapter !== null && activeChapter !== id}>{children}</div>;
}

export function ChapterReader({ explanation, children }: { explanation: Explanation; children: ReactNode }) {
  // A new result starts a new reading session, even when the result component stays mounted.
  return <Reader key={JSON.stringify(explanation)} explanation={explanation}>{children}</Reader>;
}

function Reader({ explanation, children }: { explanation: Explanation; children: ReactNode }) {
  const chapters = explanationChapters(explanation);
  const [chapterMode, setChapterMode] = useState(() => prefersChapters(explanation));
  const [index, setIndex] = useState(0);
  const heading = useRef<HTMLHeadingElement>(null);
  const headingId = useId();
  const chapter = chapters[index];

  function navigate(next: number) {
    setIndex(Math.max(0, Math.min(next, chapters.length - 1)));
    heading.current?.focus();
    heading.current?.scrollIntoView({ block: "nearest" });
  }

  return <div className="chapter-reader">
    <div className="chapter-toolbar" role="group" aria-label="Explanation reading mode">
      <span>Read at your pace</span>
      <button type="button" aria-pressed={chapterMode} onClick={() => setChapterMode(true)}>Chapters</button>
      <button type="button" aria-pressed={!chapterMode} onClick={() => setChapterMode(false)}>Full explanation</button>
    </div>
    <div className={chapterMode ? "chapter-layout" : ""}>
      {chapterMode && <nav className="chapter-sidebar" aria-label="Explanation chapters">
        <div className="section-caption">CONTENTS</div>
        <ol>{chapters.map((item, itemIndex) => <li key={item.id}>
          <button type="button" aria-current={index === itemIndex ? "step" : undefined} onClick={() => navigate(itemIndex)}><span>{itemIndex + 1}</span>{item.title}</button>
        </li>)}</ol>
      </nav>}
      <div className="chapter-body" role={chapterMode ? "region" : undefined} aria-labelledby={chapterMode ? headingId : undefined}>
        {chapterMode && <header className="chapter-heading">
          <p aria-live="polite" aria-atomic="true">Chapter {index + 1} of {chapters.length}</p>
          <h3 id={headingId} ref={heading} tabIndex={-1}>{chapter.title}</h3>
        </header>}
        <ChapterContext value={chapterMode ? chapter.id : null}>{children}</ChapterContext>
        {chapterMode && <nav className="chapter-pagination" aria-label="Chapter navigation">
          <button type="button" disabled={index === 0} onClick={() => navigate(index - 1)}>Previous</button>
          <span>{index + 1} / {chapters.length}</span>
          <button type="button" disabled={index === chapters.length - 1} onClick={() => navigate(index + 1)}>Next</button>
        </nav>}
      </div>
    </div>
  </div>;
}
