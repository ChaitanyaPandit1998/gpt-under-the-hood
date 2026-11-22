import { useLayoutEffect, useRef } from "react";

export function ChapterOutline({ label, chapters, index, onSelect }: {
  label: string; chapters: readonly { id: string; title: string }[]; index: number; onSelect: (index: number) => void;
}) {
  const list = useRef<HTMLOListElement>(null);
  const active = useRef<HTMLButtonElement>(null);
  useLayoutEffect(() => {
    const outline = list.current;
    const chapter = active.current;
    if (!outline || !chapter) return;
    const viewport = outline.getBoundingClientRect();
    const item = chapter.getBoundingClientRect();
    // Scroll only the outline; scrollIntoView can also move the reader.
    if (item.top < viewport.top) outline.scrollTop = chapter.offsetTop;
    else if (item.bottom > viewport.bottom) outline.scrollTop = chapter.offsetTop + chapter.offsetHeight - outline.clientHeight;
  }, [index]);

  return <nav className="chapter-sidebar" aria-label={label}>
    <div className="section-caption">CONTENTS</div>
    <ol ref={list}>{chapters.map((chapter, position) => <li key={chapter.id}>
      <button ref={position === index ? active : undefined} type="button" aria-current={position === index ? "step" : undefined}
        onClick={() => onSelect(position)}><span>{position + 1}</span>{chapter.title}</button>
    </li>)}</ol>
  </nav>;
}
