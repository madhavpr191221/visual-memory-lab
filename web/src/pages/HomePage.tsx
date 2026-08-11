import { FormEvent, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { AnalysisResponse, Capabilities, SearchResponse } from "../types";
import styles from "./pages.module.css";

const EXAMPLES = [
  "Where is the workstation beside a window?",
  "What is around the bookshelf workstation?",
  "Show the central aisle between desks",
  "Which memories give a clear view of the monitors?",
  "Does the aisle appear obstructed?",
];

export function HomePage() {
  const [mode, setMode] = useState<"text" | "image">("text");
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [file, setFile] = useState<File | null>(null);
  const [displayK, setDisplayK] = useState<3 | 5 | 10>(5);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.capabilities().then(setCapabilities).catch(() => setCapabilities(null));
  }, []);

  const visibleEvidence = useMemo(
    () => result?.evidence.slice(0, result.display_k) ?? [],
    [result],
  );

  async function search(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setAnalysis(null);
    setShowConfirm(false);
    try {
      const response = mode === "text"
        ? await api.searchText(question, displayK)
        : file
          ? await api.searchImage(file, displayK)
          : null;
      if (!response) throw new Error("Choose an image before searching.");
      setResult(response);
      setSelected(new Set(response.evidence.slice(0, Math.min(5, displayK)).map((item) => item.observation_id)));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  function toggleEvidence(id: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else if (next.size < 5) next.add(id);
      return next;
    });
  }

  async function runAnalysis() {
    setAnalyzing(true);
    setError("");
    try {
      const ids = [...selected];
      const response = mode === "image" && file
        ? await api.analyzeImage(question, ids, file)
        : await api.analyzeText(question, ids);
      setAnalysis(response);
      setShowConfirm(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Evidence analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <>
      <section className={styles.hero}>
        <div>
          <p className="eyebrow">Ask your office memory</p>
          <h1>Find the evidence you remember only vaguely.</h1>
          <p>Ask about a place, object, workstation, visible condition, or earlier view. The system retrieves real images first; any AI judgment remains tied to evidence you can inspect.</p>
        </div>
        <aside className={styles.heroNote}>
          <strong>What this demo knows</strong>
          <p>6,000 stored Office memories with place zones and camera pose. The public recordings do not contain calendar time or person identity.</p>
        </aside>
      </section>

      <form className={`panel ${styles.searchPanel}`} onSubmit={search}>
        <div className={styles.modeRow}>
          <div className={styles.segments} aria-label="Query type">
            <button type="button" className={mode === "text" ? styles.active : ""} onClick={() => setMode("text")}>Ask with text</button>
            <button type="button" className={mode === "image" ? styles.active : ""} onClick={() => setMode("image")}>Use an image</button>
          </div>
          <label className={styles.countSelect}>Show
            <select value={displayK} onChange={(event) => setDisplayK(Number(event.target.value) as 3 | 5 | 10)}>
              <option value={3}>3 memories</option><option value={5}>5 memories</option><option value={10}>10 memories</option>
            </select>
          </label>
        </div>
        {mode === "image" && (
          <label className={styles.upload}>Choose a PNG or JPEG under 10 MB
            <input aria-label="Query image" type="file" accept="image/png,image/jpeg" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
          </label>
        )}
        <div className={styles.question}>
          <textarea aria-label="Office memory question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What do you want to remember?" />
          <button className={styles.primary} disabled={loading || (mode === "image" && !file)}>{loading ? "Searching…" : "Find evidence"}</button>
        </div>
        <div className={styles.examples} aria-label="Example questions">
          {EXAMPLES.map((example) => <button type="button" key={example} onClick={() => { setMode("text"); setQuestion(example); }}>{example}</button>)}
        </div>
        {error && <p className="error" role="alert">{error}</p>}
      </form>

      {result && (
        <section className={styles.results} aria-live="polite">
          {result.likely_area && (
            <div className={`panel ${styles.answerStrip}`}>
              <div>
                <span className={`status ${result.likely_area.strength}`}>{result.likely_area.strength} agreement</span>
                <h2>{result.likely_area.name}</h2>
                <p>{result.likely_area.support_count} of the top {result.likely_area.evidence_count} memories point to this area.</p>
              </div>
              <div className={styles.timeNotice}>{result.temporal.message}</div>
            </div>
          )}
          <div className={styles.resultHeader}><h2>Visual evidence</h2><span>Select up to five images for judgment</span></div>
          <div className={styles.evidenceGrid}>
            {visibleEvidence.map((item) => (
              <article className={`panel ${styles.evidenceCard}`} id={`evidence-${item.observation_id}`} key={item.observation_id}>
                <label className={styles.selectEvidence} title="Select evidence">
                  <input type="checkbox" aria-label={`Select ${item.observation_id}`} checked={selected.has(item.observation_id)} onChange={() => toggleEvidence(item.observation_id)} />
                </label>
                <img src={item.image_url} alt={`Office memory ${item.rank}, ${item.zone?.name ?? "unassigned area"}`} />
                <div className={styles.evidenceBody}>
                  <div className={styles.evidenceTop}><span className={styles.rank}>#{item.rank}</span><span className={styles.score}>CLIP {item.score.toFixed(3)}</span></div>
                  <strong>{item.zone?.name ?? "Unassigned area"}</strong>
                  <small>{item.sequence_id} · frame {item.frame}</small>
                </div>
              </article>
            ))}
          </div>
          <div className={`panel ${styles.analysisBar}`}>
            <p>{selected.size} selected. Local retrieval is complete; analysis is a separate cloud action.</p>
            <button className={styles.secondary} type="button" disabled={!capabilities?.analysis_available || selected.size === 0} onClick={() => setShowConfirm(true)}>
              {capabilities?.analysis_available ? "Analyze selected evidence" : "Analysis unavailable — API key missing"}
            </button>
          </div>
          {showConfirm && (
            <div className={styles.confirm} role="dialog" aria-label="Confirm cloud analysis">
              <strong>Send selected evidence to OpenAI?</strong>
              <p>The question and {selected.size} selected image{selected.size === 1 ? "" : "s"}{mode === "image" ? ", including your uploaded query image," : ""} will be sent to <code>gpt-5.6-terra</code>. The model must cite the supplied evidence and may abstain.</p>
              <div className={styles.confirmActions}>
                <button type="button" disabled={analyzing} onClick={runAnalysis}>{analyzing ? "Analyzing…" : "Confirm and analyze"}</button>
                <button type="button" onClick={() => setShowConfirm(false)}>Cancel</button>
              </div>
            </div>
          )}
          {analysis && (
            <article className={`panel ${styles.analysis}`}>
              <span className={`status ${analysis.evidence_strength === "high" ? "strong" : analysis.evidence_strength === "medium" ? "moderate" : "mixed"}`}>{analysis.evidence_strength} evidence strength</span>
              <h2>{analysis.supported ? "Evidence-grounded answer" : "The evidence cannot support this question"}</h2>
              <p>{analysis.answer}</p>
              <div className={styles.citationList}>
                {analysis.evidence_citations.map((citation) => (
                  <a className={styles.citation} href={`#evidence-${citation.observation_id}`} key={`${citation.observation_id}-${citation.claim}`}>
                    <code>{citation.observation_id}</code>{citation.claim}
                  </a>
                ))}
              </div>
              {analysis.limitations.length > 0 && <p><strong>Limits:</strong> {analysis.limitations.join(" ")}</p>}
              <small>{analysis.model}{analysis.cached ? " · cached public-data judgment" : ""}</small>
            </article>
          )}
        </section>
      )}
    </>
  );
}
