import { useState } from "react";

type Question = { prompt: string; options: string[]; correct_index: number; explanation: string };
type Quiz = { questions: Question[] };

export function UnderstandingCheck({ source, explanation }: { source: string; explanation: unknown }) {
  const [quiz, setQuiz] = useState<Quiz | null>(null);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function createQuiz() {
    setLoading(true);
    setError("");
    setQuiz(null);
    setAnswers({});
    try {
      const response = await fetch("/api/quiz", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_text: source, explanation }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not create a check right now.");
      setQuiz(payload as Quiz);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create a check right now.");
    } finally {
      setLoading(false);
    }
  }

  return <section className="understanding-check" aria-labelledby="check-heading">
    <div className="section-caption">CHECK YOUR UNDERSTANDING</div>
    <div className="check-heading-row"><div><h3 id="check-heading">Can you predict what happens?</h3><p>Answer a few questions based on the material you shared.</p></div>
      <button className="check-button" type="button" onClick={() => void createQuiz()} disabled={loading}>{loading ? "Creating…" : quiz ? "Try another set" : "Start a quick check"}</button>
    </div>
    {loading && <p className="check-status" role="status">Creating questions from this explanation…</p>}
    {error && <p className="followup-error" role="alert">{error}</p>}
    {quiz && <div className="quiz-questions" aria-live="polite">{quiz.questions.map((question, questionIndex) => {
      const answer = answers[questionIndex];
      const answered = answer !== undefined;
      return <fieldset className="quiz-question" key={`${question.prompt}-${questionIndex}`}>
        <legend>{questionIndex + 1}. {question.prompt}</legend>
        <div className="quiz-options">{question.options.map((option, optionIndex) => <label className={`quiz-option ${answered && optionIndex === question.correct_index ? "correct" : ""} ${answered && answer === optionIndex && answer !== question.correct_index ? "incorrect" : ""}`} key={`${option}-${optionIndex}`}>
          <input type="radio" name={`quiz-${questionIndex}`} value={optionIndex} checked={answer === optionIndex} disabled={answered} onChange={() => setAnswers((current) => ({ ...current, [questionIndex]: optionIndex }))} />
          <span>{option}</span>
        </label>)}</div>
        {answered && <p className={`quiz-feedback ${answer === question.correct_index ? "is-correct" : "is-incorrect"}`} role="status">{answer === question.correct_index ? "Correct. " : "Not quite. "}{question.explanation}</p>}
      </fieldset>;
    })}</div>}
    {quiz && Object.keys(answers).length === quiz.questions.length && <p className="quiz-score" role="status">You got {Object.entries(answers).filter(([index, answer]) => quiz.questions[Number(index)]?.correct_index === answer).length} of {quiz.questions.length} correct.</p>}
  </section>;
}
