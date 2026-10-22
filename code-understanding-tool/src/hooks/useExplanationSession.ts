import { useRef, useState, type FormEvent } from "react";
import type { Explanation, ExplanationDepth, FollowUpAnswer, FollowUpMessage, LearnerLevel, Mode } from "../domain/explanation-types";

type SessionInput = { source: string; mode: Mode; depth: ExplanationDepth; learnerLevel: LearnerLevel };

export function useExplanationSession({ source, mode, depth, learnerLevel }: SessionInput) {
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isAskingFollowUp, setIsAskingFollowUp] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const [followUpQuestion, setFollowUpQuestion] = useState("");
  const [selectedTextForQuestion, setSelectedTextForQuestion] = useState("");
  const [followUpMessages, setFollowUpMessages] = useState<FollowUpMessage[]>([]);
  const [followUpError, setFollowUpError] = useState("");
  const [error, setError] = useState("");
  const explanationRequestId = useRef(0);
  const followUpRequestId = useRef(0);

  function invalidatePendingRequests() {
    explanationRequestId.current += 1;
    followUpRequestId.current += 1;
    setIsGenerating(false);
    setIsAskingFollowUp(false);
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
    setIsGenerating(true);
    setError("");
    setExplanation(null);
    setFollowUpMessages([]);
    setFollowUpQuestion("");
    setSelectedTextForQuestion("");
    setFollowUpError("");
    try {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_text: source, material_kind: mode, teaching_mode: "auto", visual_preference: "auto", explanation_depth: depth, learner_level: learnerLevel }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not generate an explanation.");
      if (requestId !== explanationRequestId.current) return;
      setExplanation(payload as Explanation);
    } catch (caught) {
      if (requestId !== explanationRequestId.current) return;
      setError(caught instanceof Error ? caught.message : "Unable to reach the local explanation service.");
    } finally {
      if (requestId === explanationRequestId.current) setIsGenerating(false);
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
        body: JSON.stringify({ source_text: source, explanation, history: requestHistory, selected_text: selectedText || undefined, question }),
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
    explanation, isGenerating, isAskingFollowUp, copyStatus, setCopyStatus,
    followUpQuestion, setFollowUpQuestion, selectedTextForQuestion, setSelectedTextForQuestion,
    followUpMessages, followUpError, error, setError,
    invalidatePendingRequests, resetExplanation, explainSource, askFollowUp,
  };
}
