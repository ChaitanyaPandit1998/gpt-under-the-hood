import { useEffect, useId, useRef, useState } from "react";
import { chapterStoryContextLimit, validateChapterStory, type ChapterStory as Story, type StoryPreferences } from "../domain/chapter-story";
import type { Mode } from "../domain/explanation-types";
import { formatMathText } from "../domain/format-math-text";
import type { SourceRange } from "../domain/source-ranges";
import { SourceRanges } from "./SourceRanges";
import { StorySceneVisual } from "./StorySceneVisual";
import type { RenderedIllustration, StorySceneImages } from "../domain/saved-study";
import type { SceneVisualPurpose } from "../domain/visual-brief";

export type ChapterStoryContext = {
  source: string;
  mode: Mode;
  preferences: StoryPreferences;
  onSelectRange: (range: SourceRange, trigger: HTMLButtonElement) => void;
  onClearCitation: () => void;
};

export function ChapterStory({ context, chapter, selected, story, images, onSelect, onReady, onRender }: {
  context: ChapterStoryContext; chapter: string; selected: boolean; story?: Story;
  images?: Record<number, StorySceneImages>;
  onSelect: (selected: boolean) => void; onReady: (story: Story) => void;
  onRender: (sceneIndex: number, purpose: SceneVisualPurpose, image: RenderedIllustration, reviewOnly?: boolean) => void;
}) {
  const request = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const panelId = useId();
  const headingId = useId();
  const lineCount = context.source.split(/\r\n|\r|\n/).length;
  useEffect(() => () => {
    request.current?.abort();
    request.current = null;
  }, []);

  function showDirect() {
    request.current?.abort();
    request.current = null;
    setBusy(false);
    setError("");
    onSelect(false);
  }

  async function showStory() {
    onSelect(true);
    if (story || request.current) return;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/chapter-story", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({
          source: context.source, mode: context.mode, ...context.preferences,
          context: chapter.length <= chapterStoryContextLimit ? chapter : `Excerpt (truncated): ${chapter.slice(0, chapterStoryContextLimit - 32)}`,
        }),
      });
      const payload = await response.json() as { story?: unknown; error?: string };
      if (!response.ok) throw new Error(payload.error || "Could not prepare a reviewed story.");
      if (request.current === controller) onReady(validateChapterStory(payload.story, lineCount));
    } catch (caught) {
      if (request.current === controller) setError(caught instanceof Error ? caught.message : "Could not prepare a reviewed story.");
    } finally {
      if (request.current === controller) {
        request.current = null;
        setBusy(false);
      }
    }
  }

  return <div className="chapter-story">
    <div className="chapter-toolbar" role="group" aria-label="Chapter explanation style">
      <span>Explain this chapter</span>
      <button type="button" aria-pressed={!selected} onClick={showDirect}>Direct</button>
      <button type="button" aria-pressed={selected} aria-controls={panelId} onClick={() => void showStory()}>Story</button>
    </div>
    <div id={panelId} hidden={!selected}>
      {busy && <div className="chapter-story-feedback">
        <p role="status">Writing and checking the story. You can keep reading the direct chapter below.</p>
        <button type="button" onClick={showDirect}>Cancel story</button>
      </div>}
      {error && <div className="chapter-story-feedback">
        <p role="alert">{error}</p>
        <button type="button" onClick={() => void showStory()}>Retry story</button>
      </div>}
      {story && selected && <article className="chapter-story-content" aria-labelledby={headingId}>
        <div className="section-caption">Story mode · Illustrative analogy</div>
        <h4 id={headingId}>{story.title}</h4>
        <p>{formatMathText(story.setup)}</p>
        <ol className="chapter-story-scenes">{story.scenes.map((scene, index) => <li key={index}>
          <h5>{scene.heading}</h5>
          <p>{formatMathText(scene.narrative)}</p>
          <div className="chapter-story-evidence">
            <strong>What this represents</strong>
            <p>{formatMathText(scene.meaning)}</p>
            <SourceRanges ranges={scene.source_ranges} lineCount={lineCount} onSelectRange={context.onSelectRange} />
          </div>
          <StorySceneVisual scene={scene} sceneNumber={index + 1} source={context.source} audience={context.preferences.learnerLevel}
            images={images?.[index]} onRender={(purpose, image, reviewOnly) => onRender(index, purpose, image, reviewOnly)} />
        </li>)}</ol>
        <p><strong>Takeaway: </strong>{formatMathText(story.takeaway)}</p>
      </article>}
    </div>
  </div>;
}
