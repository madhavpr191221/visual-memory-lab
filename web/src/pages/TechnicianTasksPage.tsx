import { useEffect, useState } from "react";
import { api } from "../api";
import type { TechnicianBenchmark } from "../types";
import styles from "./pages.module.css";

export function TechnicianTasksPage() {
  const [data, setData] = useState<TechnicianBenchmark | null>(null);
  const [selected, setSelected] = useState(0);
  const [error, setError] = useState("");
  useEffect(() => {
    api.technicianBenchmark().then(setData).catch((reason: Error) => setError(reason.message));
  }, []);
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p>Loading technician questions...</p>;
  const item = data.questions[selected];
  const summary = data.summary;
  return <>
    <header className="page-heading">
      <span className="eyebrow">Self-guided application check</span>
      <h1>Technician-style questions</h1>
      <p>Choose a practical office question, run it in the application, and inspect whether the evidence supports the result.</p>
    </header>
    <section className="panel">
      <label>Question<select value={selected} onChange={(event) => setSelected(Number(event.target.value))}>
        {data.questions.map((question, index) => <option value={index} key={question.question_id}>{question.question_id} · {question.question}</option>)}
      </select></label>
      {item && <div className={styles.taskCard}>
        <span className="eyebrow">{item.category} · {item.dataset}</span>
        <h2>{item.question}</h2>
        <p><strong>Starting observation:</strong> {item.source_observation_id}</p>
        <p>Run the case in the application first. The benchmark expectation is shown here for review, not used to alter retrieval.</p>
        <details><summary>Show benchmark expectation</summary><p><strong>Expected handling:</strong> {item.answerability.replaceAll("_", " ")}</p><p>{item.rationale}</p></details>
        <a className={styles.modeAction} href="/app">Open the application →</a>
      </div>}
    </section>
    <section className="panel">
      <h2>Benchmark status</h2>
      {summary ? <p>{String(summary.question_count)} questions evaluated. Evidence recall: {String(summary.evidence_recall ?? "not available")}.</p> : <p>No offline benchmark output yet. Run the evaluator, then return here to review the summary.</p>}
    </section>
  </>;
}
