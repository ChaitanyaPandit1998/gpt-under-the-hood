import { useId, useState } from "react";
import type { ChapterStory } from "../domain/chapter-story";
import type { Explanation, LearnerLevel } from "../domain/explanation-types";
import type { RenderedIllustration, StorySceneImages } from "../domain/saved-study";
import { defaultVisualPreferences, type SceneVisualPurpose } from "../domain/visual-brief";
import { InlineIllustration } from "./InlineIllustration";
import { VisualBriefPanel } from "./VisualBriefPanel";

const purposes = ["story-illustration", "factual-diagram"] as const;
const labels = { "story-illustration": "Story illustration", "factual-diagram": "Factual diagram" };
const figureLabels = { "story-illustration": "Story illustration: fictional analogy", "factual-diagram": "Factual diagram: generated schematic" };

function visualMaterial(scene: ChapterStory["scenes"][number], purpose: SceneVisualPurpose, sceneNumber: number): Explanation {
  const factual = purpose === "factual-diagram";
  return {
    title: factual ? `Factual mapping for scene ${sceneNumber}` : scene.heading,
    summary: factual ? scene.meaning : scene.narrative,
    concepts: [], visuals: [], assumptions: [], visual_note: "",
    walkthrough: factual ? [] : [{ heading: "What this represents", explanation: scene.meaning, source_ranges: scene.source_ranges }],
  };
}

export function StorySceneVisual({ scene, sceneNumber, source, audience, images, onRender }: {
  scene: ChapterStory["scenes"][number]; sceneNumber: number; source: string;
  audience: LearnerLevel; images?: StorySceneImages;
  onRender: (purpose: SceneVisualPurpose, image: RenderedIllustration, reviewOnly?: boolean) => void;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [purpose, setPurpose] = useState<SceneVisualPurpose>();
  const imageId = (kind: SceneVisualPurpose) => `${id}-${kind}`;
  return <div className="story-scene-visual">
    <details className="story-scene-options" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Illustrate this scene</summary>
      <div className="chapter-toolbar" role="group" aria-label="Scene image type">
        {purposes.map((kind) => <button type="button" key={kind} aria-pressed={purpose === kind} onClick={() => setPurpose(kind)}>{labels[kind]}</button>)}
      </div>
      {!purpose && <p>Choose a fictional illustration for the narrative or a diagram of the factual mapping.</p>}
      {open && purpose && <VisualBriefPanel key={purpose} embedded label={labels[purpose]} purpose={purpose}
        initialPreferences={{
          ...defaultVisualPreferences, audience, target: "screen",
          form: purpose === "story-illustration" ? "conceptual-illustration" : "auto",
          style: purpose === "story-illustration" ? "expressive" : "technical",
        }}
        explanation={visualMaterial(scene, purpose, sceneNumber)} source={source} scope={`${labels[purpose]} for scene ${sceneNumber}`}
        placement={{ id: imageId(purpose), current: images?.[purpose], location: "scene",
          onRender: (image) => onRender(purpose, image), onReview: (image) => onRender(purpose, image, true) }} />}
    </details>
    {purposes.map((kind) => {
      const image = images?.[kind];
      return image && <InlineIllustration key={kind} id={imageId(kind)} image={image} label={figureLabels[kind]} onChange={updated => onRender(kind, updated, true)} />;
    })}
  </div>;
}
