import { createContext, useContext, useId, useImperativeHandle, useRef, useState, type ReactNode, type Ref } from "react";
import type { Explanation } from "../domain/explanation-types";
import { explanationChapters, prefersChapters, type ExplanationChapter } from "../domain/explanation-chapters";
import { ChapterIllustration } from "./ChapterIllustration";
import { ChapterOutline } from "./ChapterOutline";
import type { ChapterArtwork } from "../domain/chapter-illustrations";
import { ChapterRefinement, type ChapterRefinementContext, type ChapterRefinementHandle } from "./ChapterRefinement";
import { VisualBriefPanel } from "./VisualBriefPanel";
import { InlineIllustration } from "./InlineIllustration";
import { ReadingTools } from "./ReadingTools";
import type { ReadingState, RenderedIllustration } from "../domain/saved-study";
import type { ChapterStory as Story } from "../domain/chapter-story";
import { ChapterStory, type ChapterStoryContext } from "./ChapterStory";
import type { StorySceneImages } from "./StorySceneVisual";

const ChapterContext = createContext<{
  active: string | null; imagePrefix: string; images: Record<string, RenderedIllustration>;
} | null>(null);

export function ChapterPage({ id, children }: { id: string; children: ReactNode }) {
  const context = useContext(ChapterContext);
  const image = context?.images[id];
  return <div className="chapter-page" hidden={!!context?.active && context.active !== id}>
    {children}
    {context && image && <InlineIllustration id={`${context.imagePrefix}-${id}`} image={image} />}
  </div>;
}

function chapterMaterial(explanation: Explanation, chapter: ExplanationChapter): Explanation {
  const { id, title } = chapter;
  const step = id.startsWith("step-") ? explanation.walkthrough[Number(id.slice(5))] : undefined;
  const summary = id === "overview" ? explanation.summary : step?.explanation ||
    (id === "visuals" ? explanation.visual_note : "") || `${title} in ${explanation.title}`;
  return {
    title: `${explanation.title}: ${title}`, summary,
    concepts: id === "concepts" ? explanation.concepts : [],
    walkthrough: step ? [step] : [],
    visuals: id === "visuals" ? explanation.visuals : [],
    visual_note: id === "visuals" ? explanation.visual_note : "",
    assumptions: explanation.assumptions,
    incomplete_items: explanation.incomplete_items,
    quality_review_status: explanation.quality_review_status,
    quality_review_note: explanation.quality_review_note,
  };
}

export type ChapterReaderHandle = { snapshot: () => ReadingState };
type ReaderProps = { explanation: Explanation; children: ReactNode; artworkEnabled?: boolean; refinementContext?: ChapterRefinementContext; storyContext?: ChapterStoryContext; initialReading?: ReadingState; ref?: Ref<ChapterReaderHandle> };

export function ChapterReader({ explanation, ...props }: ReaderProps) {
  // A new result starts a new reading session, even when the result component stays mounted.
  return <Reader key={JSON.stringify(explanation)} explanation={explanation} {...props} />;
}

function Reader({ explanation, children, artworkEnabled = true, refinementContext, storyContext, initialReading, ref }: ReaderProps) {
  const chapters = explanationChapters(explanation);
  const [chapterMode, setChapterMode] = useState(() => initialReading?.chapterMode ?? prefersChapters(explanation));
  const [index, setIndex] = useState(() => Math.max(0, chapters.findIndex((chapter) => chapter.id === initialReading?.chapterId)));
  const [covers, setCovers] = useState<Record<string, ChapterArtwork>>(() => initialReading?.covers ?? {});
  const [images, setImages] = useState<Record<string, RenderedIllustration>>(() => initialReading?.images ?? {});
  const [stories, setStories] = useState<Record<string, Story>>({});
  const [storyImages, setStoryImages] = useState<Record<string, Record<number, StorySceneImages>>>({});
  const [selectedStory, setSelectedStory] = useState<string>();
  const heading = useRef<HTMLHeadingElement>(null);
  const refinementReaders = useRef(new Map<string, ChapterRefinementHandle>());
  const headingId = useId();
  const imagePrefix = useId();
  const chapter = chapters[index];
  const chapterContext = chapterMode && (refinementContext || storyContext) ? chapterMaterial(explanation, chapter) : undefined;
  const storyKey = JSON.stringify([chapter.id, storyContext?.preferences]);
  const storySelected = selectedStory === storyKey;
  const storyVisible = chapterMode && storySelected && !!stories[storyKey];
  useImperativeHandle(ref, () => ({ snapshot: () => ({
    chapterMode, chapterId: chapter.id, images, covers,
    adapted: Object.fromEntries([...refinementReaders.current].map(([id, refinement]) => [id, refinement.snapshot()])),
  }) }), [chapterMode, chapter.id, images, covers]);

  function navigate(next: number) {
    if (next !== index && storySelected) {
      setSelectedStory(undefined);
      storyContext?.onClearCitation();
    }
    setIndex(Math.max(0, Math.min(next, chapters.length - 1)));
    heading.current?.focus();
    heading.current?.scrollIntoView({ block: "nearest" });
  }

  return <div className="chapter-reader">
    <div className="chapter-toolbar" role="group" aria-label="Explanation reading mode">
      <span>Read at your pace</span>
      <button type="button" aria-pressed={chapterMode} onClick={() => setChapterMode(true)}>Chapters</button>
      <button type="button" aria-pressed={!chapterMode} onClick={() => {
        if (storySelected) storyContext?.onClearCitation();
        setSelectedStory(undefined);
        setChapterMode(false);
      }}>Full explanation</button>
    </div>
    <div className={chapterMode ? "chapter-layout" : ""}>
      {chapterMode && <ChapterOutline label="Explanation chapters" chapters={chapters} index={index} onSelect={navigate} />}
      <div className="chapter-body" role={chapterMode ? "region" : undefined} aria-labelledby={chapterMode ? headingId : undefined}>
        {chapterMode && <header className="chapter-heading">
          <p aria-live="polite" aria-atomic="true">Chapter {index + 1} of {chapters.length}</p>
          <h3 id={headingId} ref={heading} tabIndex={-1}>{chapter.title}</h3>
        </header>}
        {chapterContext && storyContext?.source.trim() && <ChapterStory key={storyKey} context={storyContext} chapter={JSON.stringify(chapterContext)}
          selected={storySelected} story={stories[storyKey]} images={storyImages[storyKey]} onSelect={(selected) => {
            if (!selected && storyVisible) storyContext.onClearCitation();
            setSelectedStory(selected ? storyKey : undefined);
          }} onReady={(story) => {
            storyContext.onClearCitation();
            setStories((current) => ({ ...current, [storyKey]: story }));
          }} onRender={(sceneIndex, purpose, image) => setStoryImages((current) => ({
            ...current, [storyKey]: { ...current[storyKey], [sceneIndex]: { ...current[storyKey]?.[sceneIndex], [purpose]: image } },
          }))} />}
        <div className="chapter-direct-content" hidden={storyVisible}>
        {chapterContext && refinementContext && <ReadingTools key={`tools-${chapter.id}`} label="Chapter tools" scope={`This chapter: ${chapter.title}`}
          explanation={explanation} source={refinementContext.source} context={{ scope: `Chapter: ${chapter.title}`, provenance: "" }}
          refine={<a href={`#${imagePrefix}-refine-${chapter.id}`}>Refine chapter</a>}
          visual={<VisualBriefPanel label="Illustrate chapter" scope={`Chapter: ${chapter.title}`}
            explanation={chapterContext} source={refinementContext.source}
            placement={{ id: `${imagePrefix}-${chapter.id}`, current: images[chapter.id], onRender: (image) => setImages((current) => ({ ...current, [chapter.id]: image })) }} />}
          chapterSpeech={{ ...chapterContext, summary: chapter.id.startsWith("step-") ? "" : chapterContext.summary,
            assumptions: chapter.id === "recap" ? chapterContext.assumptions : [], incomplete_items: [] }}
          collectImages={() => ({ chapters: { [chapter.id]: images[chapter.id] } })} onlyChapterId={chapter.id} />}
        {chapterMode && artworkEnabled && <ChapterIllustration key={`cover-${chapter.id}`} cached={covers[chapter.id]} onChange={(artwork) => setCovers((current) => {
          const next = { ...current };
          if (artwork) next[chapter.id] = artwork;
          else delete next[chapter.id];
          return next;
        })} />}
        {refinementContext && chapters.map((item) => <ChapterRefinement key={item.id} id={`${imagePrefix}-refine-${item.id}`} context={refinementContext} title={item.title} active={chapterMode && item.id === chapter.id}
          initial={initialReading?.adapted?.[item.id]} ref={(handle) => {
            if (handle) refinementReaders.current.set(item.id, handle);
            else refinementReaders.current.delete(item.id);
          }}
          section={item.id === "overview" ? explanation.summary : item.id === "concepts" ? explanation.concepts : item.id.startsWith("step-") ? explanation.walkthrough[Number(item.id.slice(5))] : item.id === "visuals" ? { visuals: explanation.visuals, note: explanation.visual_note } : explanation.assumptions} />)}
        <ChapterContext value={{ active: chapterMode ? chapter.id : null, imagePrefix, images }}>{children}</ChapterContext>
        </div>
        {chapterMode && <nav className="chapter-pagination" aria-label="Chapter navigation">
          <button type="button" disabled={index === 0} onClick={() => navigate(index - 1)}>Previous</button>
          <span>{index + 1} / {chapters.length}</span>
          <button type="button" disabled={index === chapters.length - 1} onClick={() => navigate(index + 1)}>Next</button>
        </nav>}
      </div>
    </div>
  </div>;
}
