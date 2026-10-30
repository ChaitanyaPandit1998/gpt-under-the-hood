import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Explanation } from "../domain/explanation-types";

type ReadSection = "summary" | "walkthrough" | "diagram" | "full";
type PlaybackState = "idle" | "speaking" | "paused";

export function SpeechReader({ explanation }: { explanation: Explanation }) {
  const [readSection, setReadSection] = useState<ReadSection>("full");
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [voiceURI, setVoiceURI] = useState("");
  const [playbackState, setPlaybackState] = useState<PlaybackState>("idle");
  const [error, setError] = useState("");
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const speechSynthesis = typeof window !== "undefined" && "speechSynthesis" in window
    ? window.speechSynthesis
    : null;

  const textBySection = useMemo(() => {
    const summary = [explanation.title, explanation.summary].filter(Boolean).join(". ");
    const walkthrough = explanation.walkthrough
      .map((step) => `${step.heading}. ${step.explanation}`)
      .join("\n\n");
    const diagram = explanation.visuals.map((visual) => {
      const itemLabel = visual.kind === "control_flow" ? "Steps" : "Concepts";
      const nodes = visual.nodes.map((node) => {
        const incomplete = explanation.incomplete_items?.some((item) =>
          item.source_ranges.some((range) => node.source_ranges.some((nodeRange) =>
            range.start_line <= nodeRange.end_line && nodeRange.start_line <= range.end_line,
          )),
        );
        return `${incomplete ? "Incomplete implementation. " : ""}${node.label}. ${node.description}`;
      }).join(". ");
      const connections = visual.edges.map((edge) => {
        const from = visual.nodes.find((node) => node.id === edge.from)?.label ?? edge.from;
        const to = visual.nodes.find((node) => node.id === edge.to)?.label ?? edge.to;
        return `${from} ${edge.label ? `when ${edge.label}` : "leads to"} ${to}`;
      }).join(". ");
      return `${visual.title}. ${itemLabel}. ${nodes}${connections ? `. Connections. ${connections}` : ""}`;
    }).join("\n\n");
    const full = [
      summary,
      explanation.concepts.length > 0
        ? `Key ideas. ${explanation.concepts.map((concept) => `${concept.term}. ${concept.explanation}`).join(". ")}`
        : "",
      walkthrough ? `Walkthrough. ${walkthrough}` : "",
      explanation.visual_note ? `Diagram coverage note. ${explanation.visual_note}` : "",
      explanation.incomplete_items?.length
        ? `Incomplete code detected. ${explanation.incomplete_items.map((item) => `${item.title}. ${item.explanation}`).join(" ")}`
        : "",
      explanation.assumptions.length > 0
        ? `What the input does not tell us. ${explanation.assumptions.map((item) => `${item.statement}. ${item.reason}`).join(". ")}`
        : "",
      diagram,
    ].filter(Boolean).join("\n\n");

    return { summary, walkthrough, diagram, full };
  }, [explanation]);

  const activeReadSection = (readSection === "walkthrough" && !textBySection.walkthrough)
    || (readSection === "diagram" && !textBySection.diagram) ? "full" : readSection;
  const text = textBySection[activeReadSection] || textBySection.summary;
  const selectedVoice = voices.find((voice) => voice.voiceURI === voiceURI);

  useEffect(() => {
    if (!speechSynthesis) return;
    const updateVoices = () => setVoices(speechSynthesis.getVoices());
    updateVoices();
    speechSynthesis.addEventListener("voiceschanged", updateVoices);
    return () => {
      speechSynthesis.removeEventListener("voiceschanged", updateVoices);
      utteranceRef.current = null;
      speechSynthesis.cancel();
    };
  }, [speechSynthesis]);

  const stop = useCallback(() => {
    utteranceRef.current = null;
    speechSynthesis?.cancel();
    setPlaybackState("idle");
  }, [speechSynthesis]);

  useEffect(() => {
    stop();
    setError("");
  }, [text, stop]);

  if (!speechSynthesis) {
    return <p className="speech-reader-unavailable">Read aloud is not available in this browser.</p>;
  }

  function startReading() {
    if (!speechSynthesis || !text.trim()) return;
    speechSynthesis.cancel();
    setError("");
    const utterance = new SpeechSynthesisUtterance(text);
    if (selectedVoice) {
      utterance.voice = selectedVoice;
      utterance.lang = selectedVoice.lang;
    } else {
      utterance.lang = document.documentElement.lang || navigator.language;
    }
    utterance.onstart = () => {
      if (utteranceRef.current === utterance) setPlaybackState("speaking");
    };
    utterance.onend = () => {
      if (utteranceRef.current === utterance) {
        utteranceRef.current = null;
        setPlaybackState("idle");
      }
    };
    utterance.onerror = (event) => {
      if (utteranceRef.current !== utterance) return;
      utteranceRef.current = null;
      setPlaybackState("idle");
      if (event.error !== "canceled" && event.error !== "interrupted") {
        setError("The selected voice could not read this content. Try another voice.");
      }
    };
    utteranceRef.current = utterance;
    setPlaybackState("speaking");
    speechSynthesis.speak(utterance);
  }

  function togglePlayback() {
    if (!speechSynthesis) return;
    if (playbackState === "speaking") {
      speechSynthesis.pause();
      setPlaybackState("paused");
    } else if (playbackState === "paused") {
      speechSynthesis.resume();
      setPlaybackState("speaking");
    } else {
      startReading();
    }
  }

  return <div className="speech-reader" role="group" aria-label="Read explanation or diagram aloud">
    <label className="speech-reader-select">Read
      <select value={activeReadSection} onChange={(event) => setReadSection(event.target.value as ReadSection)}>
        <option value="summary">Summary</option>
        <option value="walkthrough" disabled={!textBySection.walkthrough}>Walkthrough</option>
        <option value="diagram" disabled={!textBySection.diagram}>Diagram</option>
        <option value="full">Full explanation</option>
      </select>
    </label>
    <label className="speech-reader-select">Voice
      <select value={voiceURI} onChange={(event) => setVoiceURI(event.target.value)} disabled={voices.length === 0 || playbackState !== "idle"}>
        <option value="">System default</option>
        {voices.map((voice) => <option key={voice.voiceURI} value={voice.voiceURI}>{voice.name} ({voice.lang}){voice.default ? " — default" : ""}</option>)}
      </select>
    </label>
    <button
      className="speech-reader-button speech-reader-icon-button"
      type="button"
      onClick={togglePlayback}
      aria-label={playbackState === "speaking" ? "Pause reading" : playbackState === "paused" ? "Resume reading" : `Read ${activeReadSection === "diagram" ? "diagram" : "explanation"} aloud`}
      title={playbackState === "speaking" ? "Pause reading" : playbackState === "paused" ? "Resume reading" : "Read aloud"}
    >
      {playbackState === "speaking" ? <PauseIcon /> : playbackState === "paused" ? <PlayIcon /> : <SpeakerIcon />}
    </button>
    <button className="speech-reader-stop speech-reader-icon-button" type="button" onClick={stop} disabled={playbackState === "idle"} aria-label="Stop reading" title="Stop">
      <StopIcon />
    </button>
    <span className="visually-hidden" role="status" aria-live="polite">
      {playbackState === "speaking" ? `Reading ${activeReadSection === "full" ? "the full explanation" : `the ${activeReadSection}`}.` : playbackState === "paused" ? "Reading paused." : "Reading stopped."}
    </span>
    {error && <span className="speech-reader-error" role="alert">{error}</span>}
  </div>;
}

function SpeakerIcon() {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M11 5 6 9H3v6h3l5 4V5Z" /><path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13" /></svg>;
}

function PauseIcon() {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>;
}

function PlayIcon() {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.8a1 1 0 0 1 1.5-.86l11 7.2a1 1 0 0 1 0 1.72l-11 7.2A1 1 0 0 1 7 19.2V4.8Z" /></svg>;
}

function StopIcon() {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>;
}
