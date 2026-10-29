import { useEffect, useRef, useState, type FormEvent } from "react";
import { MAX_FOLLOW_UP_HISTORY_MESSAGES, type Explanation, type ExplanationDepth, type FollowUpAnswer, type FollowUpMessage, type LearnerLevel, type Mode } from "../domain/explanation-types";
import { isGenerationPhase, type GenerationPhase } from "../domain/generation-progress";

type SessionInput = { source: string; mode: Mode; depth: ExplanationDepth; learnerLevel: LearnerLevel };

function createProgressId(): string {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (character) => {
    const random = Math.floor(Math.random() * 16);
    return (character === "x" ? random : (random & 0x3) | 0x8).toString(16);
  });
}

export function useExplanationSession({ source, mode, depth, learnerLevel }: SessionInput) {
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationPhase, setGenerationPhase] = useState<GenerationPhase | null>(null);
  const [isAskingFollowUp, setIsAskingFollowUp] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const [followUpQuestion, setFollowUpQuestion] = useState("");
  const [selectedTextForQuestion, setSelectedTextForQuestion] = useState("");
  const [followUpMessages, setFollowUpMessages] = useState<FollowUpMessage[]>([]);
  const [followUpError, setFollowUpError] = useState("");
  const [error, setError] = useState("");
  const explanationRequestId = useRef(0);
  const followUpRequestId = useRef(0);
  const explanationRequestController = useRef<AbortController | null>(null);
  const progressPollingController = useRef<AbortController | null>(null);

  useEffect(() => () => {
    explanationRequestId.current += 1;
    explanationRequestController.current?.abort();
    progressPollingController.current?.abort();
  }, []);

  function invalidatePendingRequests() {
    explanationRequestId.current += 1;
    followUpRequestId.current += 1;
    setIsGenerating(false);
    setGenerationPhase(null);
    setIsAskingFollowUp(false);
    explanationRequestController.current?.abort();
    explanationRequestController.current = null;
    progressPollingController.current?.abort();
    progressPollingController.current = null;
  }

  function resetExplanation() {
    invalidatePendingRequests();
    setExplanation(null);
    setFollowUpMessages([]);
    setFollowUpQuestion("");
    setSelectedTextForQuestion("");
    setFollowUpError("");
    setError("");
  }

  async function explainSource() {
    const requestId = ++explanationRequestId.current;
    const progressId = createProgressId();
    const requestController = new AbortController();
    const progressController = new AbortController();
    explanationRequestController.current?.abort();
    explanationRequestController.current = requestController;
    progressPollingController.current?.abort();
    progressPollingController.current = progressController;
    setIsGenerating(true);
    setGenerationPhase("preparing");
    setError("");
    setExplanation(null);
    setFollowUpMessages([]);
    setFollowUpQuestion("");
    setSelectedTextForQuestion("");
    setFollowUpError("");
    let polling = true;
    const pollProgress = async () => {
      while (polling && requestId === explanationRequestId.current) {
        try {
          const progressResponse = await fetch(`/api/explain-progress?id=${encodeURIComponent(progressId)}`, { signal: progressController.signal });
          if (progressResponse.ok) {
            const progressPayload = await progressResponse.json() as { phase?: unknown };
            if (isGenerationPhase(progressPayload.phase) && requestId === explanationRequestId.current) {
              setGenerationPhase(progressPayload.phase);
            }
          }
        } catch {
          if (progressController.signal.aborted) return;
          // Progress is best-effort; the explanation request still reports success or failure.
        }
        await new Promise((resolve) => window.setTimeout(resolve, 700));
      }
    };
    const progressPolling = pollProgress();
    try {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_text: source, material_kind: mode, teaching_mode: "auto", visual_preference: "auto", explanation_depth: depth, learner_level: learnerLevel, progress_id: progressId }),
        signal: requestController.signal,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not generate an explanation.");
      if (requestId !== explanationRequestId.current) return;
      setExplanation(payload as Explanation);
    } catch (caught) {
      if (requestId !== explanationRequestId.current) return;
      setError(caught instanceof Error ? caught.message : "Unable to reach the local explanation service.");
    } finally {
      polling = false;
      if (explanationRequestController.current === requestController) explanationRequestController.current = null;
      progressController.abort();
      if (progressPollingController.current === progressController) progressPollingController.current = null;
      if (requestId === explanationRequestId.current) {
        setIsGenerating(false);
        setGenerationPhase(null);
      }
      await progressPolling;
    }
  }

  async function askFollowUp(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = followUpQuestion.trim();
    if (!question || !explanation || isAskingFollowUp) return;
    const requestId = ++followUpRequestId.current;
    const requestHistory = followUpMessages;
    const selectedText = selectedTextForQuestion;
    setIsAskingFollowUp(true);
    setFollowUpError("");
    try {
      const response = await fetch("/api/followup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_text: source, explanation, history: requestHistory.slice(-MAX_FOLLOW_UP_HISTORY_MESSAGES), selected_text: selectedText || undefined, question }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not answer this follow-up.");
      if (requestId !== followUpRequestId.current) return;
      const answer = payload as FollowUpAnswer;
      setFollowUpMessages((current) => [
        ...current,
        { role: "learner", text: question, ...(selectedText ? { selected_text: selectedText } : {}) },
        { role: "assistant", text: answer.answer, source_ranges: answer.source_ranges, assumptions: answer.assumptions },
      ]);
      setFollowUpQuestion("");
      setSelectedTextForQuestion("");
    } catch (caught) {
      if (requestId !== followUpRequestId.current) return;
      setFollowUpError(caught instanceof Error ? caught.message : "Unable to reach the local explanation service.");
    } finally {
      if (requestId === followUpRequestId.current) setIsAskingFollowUp(false);
    }
  }

  return {
    explanation, isGenerating, generationPhase, isAskingFollowUp, copyStatus, setCopyStatus,
    followUpQuestion, setFollowUpQuestion, selectedTextForQuestion, setSelectedTextForQuestion,
    followUpMessages, followUpError, error, setError,
    invalidatePendingRequests, resetExplanation, explainSource, askFollowUp,
  };
}
