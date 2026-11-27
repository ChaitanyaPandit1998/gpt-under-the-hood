import { useEffect, useRef, useState, type FormEvent } from "react";
import { MAX_FOLLOW_UP_HISTORY_MESSAGES, type Explanation, type ExplanationDepth, type FollowUpAnswer, type FollowUpMessage, type LearnerLevel, type Mode } from "../domain/explanation-types";
import { isGenerationPhase, type GenerationPhase } from "../domain/generation-progress";
import type { ChapterSettings } from "../domain/chapter-settings";

type SessionInput = { source: string; mode: Mode; depth: ExplanationDepth; learnerLevel: LearnerLevel; tone?: ChapterSettings["tone"]; focus?: ChapterSettings["focus"]; visualPreference?: ChapterSettings["visualPreference"]; learnerQuestion?: string };

function createProgressId(): string {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (character) => {
    const random = Math.floor(Math.random() * 16);
    return (character === "x" ? random : (random & 0x3) | 0x8).toString(16);
  });
}

export function useExplanationSession({ source, mode, depth, learnerLevel, tone = "plain", focus = "auto", visualPreference = "auto", learnerQuestion }: SessionInput, initialExplanation: Explanation | null = null) {
  const [explanation, setExplanation] = useState<Explanation | null>(initialExplanation);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isReviewing, setIsReviewing] = useState(false);
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
  const followUpRequestController = useRef<AbortController | null>(null);
  const progressPollingController = useRef<AbortController | null>(null);

  useEffect(() => () => {
    explanationRequestId.current += 1;
    followUpRequestId.current += 1;
    explanationRequestController.current?.abort();
    followUpRequestController.current?.abort();
    progressPollingController.current?.abort();
  }, []);

  function cancelFollowUp() {
    followUpRequestId.current += 1;
    followUpRequestController.current?.abort();
    followUpRequestController.current = null;
    setIsAskingFollowUp(false);
  }

  function invalidatePendingRequests() {
    explanationRequestId.current += 1;
    cancelFollowUp();
    setIsGenerating(false);
    setIsReviewing(false);
    setGenerationPhase(null);
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

  async function explainSource(action?: "draft" | "review" | "draft-and-review") {
    if (explanationRequestController.current) return;
    if (action === "review" && !explanation) return;
    cancelFollowUp();
    const requestId = ++explanationRequestId.current;
    let progressId = createProgressId();
    const requestController = new AbortController();
    const progressController = new AbortController();
    explanationRequestController.current = requestController;
    progressPollingController.current?.abort();
    progressPollingController.current = progressController;
    setIsGenerating(true);
    setIsReviewing(action === "review");
    setGenerationPhase(action === "review" ? "reviewing" : "preparing");
    setError("");
    let polling = true;
    const pollProgress = async () => {
      while (polling && requestId === explanationRequestId.current) {
        try {
          const polledId = progressId;
          const progressResponse = await fetch(`/api/explain-progress?id=${encodeURIComponent(polledId)}`, { signal: progressController.signal });
          if (progressResponse.ok) {
            const progressPayload = await progressResponse.json() as { phase?: unknown };
            if (isGenerationPhase(progressPayload.phase) && requestId === explanationRequestId.current && polledId === progressId) {
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
    async function submit(reviewMode?: "draft" | "review", draft?: Explanation): Promise<Explanation> {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_text: source, material_kind: mode === "code" ? "code" : "docs", teaching_mode: "auto", visual_preference: visualPreference, explanation_depth: depth, learner_level: learnerLevel, explanation_tone: tone, explanation_focus: focus, learner_question: learnerQuestion, progress_id: progressId, review_mode: reviewMode, draft }),
        signal: requestController.signal,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not generate an explanation.");
      return payload as Explanation;
    }
    try {
      const payload = await (action === "review" ? submit("review", explanation!) : submit(action ? "draft" : undefined));
      if (requestId !== explanationRequestId.current) return;
      setExplanation(payload);
      setCopyStatus("");
      setFollowUpMessages([]);
      setFollowUpQuestion("");
      setSelectedTextForQuestion("");
      setFollowUpError("");
      if (action === "draft-and-review") {
        progressId = createProgressId();
        setIsReviewing(true);
        setGenerationPhase("reviewing");
        const reviewed = await submit("review", payload);
        if (requestId === explanationRequestId.current) setExplanation(reviewed);
      }
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
        setIsReviewing(false);
        setGenerationPhase(null);
      }
      await progressPolling;
    }
  }

  async function askFollowUp(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = followUpQuestion.trim();
    if (!question || !explanation || isAskingFollowUp || explanationRequestController.current) return;
    const requestId = ++followUpRequestId.current;
    const requestController = new AbortController();
    followUpRequestController.current?.abort();
    followUpRequestController.current = requestController;
    const requestHistory = followUpMessages;
    const selectedText = selectedTextForQuestion;
    setIsAskingFollowUp(true);
    setFollowUpError("");
    try {
      const response = await fetch("/api/followup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_text: source, explanation, history: requestHistory.slice(-MAX_FOLLOW_UP_HISTORY_MESSAGES), selected_text: selectedText || undefined, question }),
        signal: requestController.signal,
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
      if (followUpRequestController.current === requestController) followUpRequestController.current = null;
      if (requestId === followUpRequestId.current) setIsAskingFollowUp(false);
    }
  }

  return {
    explanation, setExplanation, isGenerating, isReviewing, generationPhase, isAskingFollowUp, copyStatus, setCopyStatus,
    followUpQuestion, setFollowUpQuestion, selectedTextForQuestion, setSelectedTextForQuestion,
    followUpMessages, followUpError, error, setError,
    invalidatePendingRequests, resetExplanation, explainSource, askFollowUp,
  };
}
