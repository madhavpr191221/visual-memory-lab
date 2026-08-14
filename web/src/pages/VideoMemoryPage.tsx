import { FormEvent, useState } from "react";
import { api } from "../api";
import type { VideoMemoryResponse } from "../types";
import styles from "./pages.module.css";

export function VideoMemoryPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<VideoMemoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function search(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    try {
      setResult(await api.videoMemory(query));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Video search failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <section className={styles.hero}>
        <div>
          <p className="eyebrow">Video memory</p>
          <h1>Find the moment, not just the video.</h1>
          <p>Ask when an action happened or when an object appeared. Each result is a short time window with the annotation and video evidence beside it.</p>
        </div>
        <aside className={styles.heroNote}>
          <strong>Charades temporal memory</strong>
          <p>The first slice uses official action intervals as a transparent baseline. Learned CLIP and temporal retrieval will replace this baseline after the workflow is validated.</p>
        </aside>
      </section>
      <form className={`panel ${styles.searchPanel}`} onSubmit={search}>
        <div className={styles.question}>
          <textarea aria-label="Video memory question" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="When did the person open the door?" />
          <button className={styles.primary} disabled={loading || !query.trim()}>{loading ? "Searching…" : "Find moment"}</button>
        </div>
        <div className={styles.examples}>
          {["When did the person open the door?", "When did the person sit down?", "When was the book visible?"].map((example) => (
            <button type="button" key={example} onClick={() => setQuery(example)}>{example}</button>
          ))}
        </div>
        {error && <p className="error" role="alert">{error}</p>}
      </form>
      {result && (
        <section className={styles.results} aria-live="polite">
          <div className={`panel ${styles.answerStrip}`}>
            <div><span className="status strong">candidate moments</span><h2>{result.results.length} candidate moments</h2><p>{result.window_count} timestamped windows are available in the prepared Charades subset.</p></div>
            <div className={styles.timeNotice}>Baseline: annotation matching; review the video evidence.</div>
          </div>
          <div className={styles.evidenceGrid}>
            {result.results.map((item) => (
              <article className={`panel ${styles.evidenceCard}`} key={item.window_id}>
                {item.video_url && <video controls preload="metadata" src={`${item.video_url}#t=${item.start_s},${item.end_s}`} />}
                <div className={styles.evidenceBody}>
                  <div className={styles.evidenceTop}><span className={styles.rank}>{item.video_id}</span><span className={styles.score}>{item.start_s.toFixed(1)}–{item.end_s.toFixed(1)} s</span></div>
                  <strong>{item.actions.map((action) => action.name).join(" · ") || "Relevant video window"}</strong>
                  <small>Video-level objects: {item.objects.join(" · ") || "none listed"}</small>
                  {item.description && <p>{item.description}</p>}
                </div>
              </article>
            ))}
          </div>
          {result.results.length === 0 && <div className="panel"><p>No annotated window matched that question. This is a safe no-result, not proof that the event never happened.</p></div>}
        </section>
      )}
    </>
  );
}
