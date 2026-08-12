import { useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../api";
import type { QueryPage } from "../types";
import styles from "./pages.module.css";

const TAGS = ["", "strict-top1", "strict-rescued-at5", "strict-rescued-at10", "strict-miss-at10", "uncovered", "relaxed-only-top1", "large-translation", "large-rotation"];
const TAG_GUIDE: Record<string, string> = {
  "strict-top1": "The first result is within 0.25 m and 30° of the query pose.",
  "strict-rescued-at5": "The first result missed, but one of the top five meets the strict pose threshold.",
  "strict-rescued-at10": "The first five missed, but one of the top ten meets the strict pose threshold.",
  "strict-miss-at10": "None of the top ten results meets the strict pose threshold, despite having usable coverage.",
  uncovered: "The reference memory has no sufficiently nearby view, so retrieval cannot be judged strictly.",
  "relaxed-only-top1": "The first result is outside the strict threshold but falls within the wider relaxed threshold.",
  "large-translation": "The first result is more than 0.50 m away from the query camera position.",
  "large-rotation": "The first result differs from the query camera direction by more than 30°.",
};

export function FailuresPage() {
  const [page, setPage] = useState<QueryPage | null>(null);
  const [offset, setOffset] = useState(0);
  const [tag, setTag] = useState("");
  const [error, setError] = useState("");
  useEffect(() => { setError(""); api.queries(offset, tag).then(setPage).catch((reason: Error) => setError(reason.message)); }, [offset, tag]);
  const changeTag = (value: string) => { setTag(value); setOffset(0); };
  return <>
    <header className="page-heading"><span className="eyebrow">Evidence Lab</span><h1>Failure browser</h1><p>Inspect the actual query frame and its nearest memories. Tags describe retrieval outcomes; they are not model-generated explanations.</p></header>
    <div className={styles.filterBar}><label>Outcome <select value={tag} onChange={(event) => changeTag(event.target.value)}>{TAGS.map((value) => <option key={value} value={value}>{value || "all queries"}</option>)}</select></label>{page && <span>{page.total.toLocaleString()} matching queries</span>}</div>
    <section className={`panel ${styles.metricGuide}`} aria-label="Metric explanations">
      <div><strong>Translation error</strong><span>Distance from the query camera position; lower is better.</span></div>
      <div><strong>Rotation error</strong><span>Difference in camera direction; lower is better.</span></div>
      <div><strong>Strict match</strong><span>Correct only within 0.25 m and 30 degrees.</span></div>
      <div><strong>Relaxed match</strong><span>Allows up to 0.50 m position error.</span></div>
      <div><strong>Coverage</strong><span>Whether a nearby reference view exists to judge.</span></div>
      {TAGS.slice(1).map((value) => <div key={value}><strong>{value}</strong><span>{TAG_GUIDE[value]}</span></div>)}
    </section>
    {error && <p className="error">{error}</p>}
    {!page ? <p>Loading queries…</p> : <>
      <section className={styles.queryGrid}>{page.items.map((item) => <Link className={`panel ${styles.queryCard}`} to={`/lab/queries/${encodeURIComponent(item.query_id)}`} key={item.query_id}><img src={item.image_url} alt={`Query ${item.query_id}`} loading="lazy" /><div className={styles.queryBody}><strong>{item.sequence_id} · frame {item.frame}</strong><p>{item.top1_translation_error_m.toFixed(2)} m · {item.top1_rotation_error_deg.toFixed(1)}°</p><div className={styles.tags}>{item.tags.map((value) => <span className={styles.tag} key={value}>{value}</span>)}</div></div></Link>)}</section>
      <nav className={styles.pager} aria-label="Query pages"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - page.limit))}>Previous</button><span>{offset + 1}–{Math.min(offset + page.limit, page.total)} of {page.total}</span><button disabled={offset + page.limit >= page.total} onClick={() => setOffset(offset + page.limit)}>Next</button></nav>
    </>}
  </>;
}
