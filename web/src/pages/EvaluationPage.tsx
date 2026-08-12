import { useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../api";
import styles from "./pages.module.css";

type Hit = { all_query_rate?: number; covered_rate?: number };
type Protocol = { coverage?: number; hit_at_1?: Hit; hit_at_5?: Hit; hit_at_10?: Hit };
type Pose = { query_count?: number; strict?: Protocol; relaxed?: Protocol; per_sequence?: Record<string, { strict_coverage?: number; strict_hit_at_1_all_queries?: number; strict_hit_at_5_all_queries?: number }>; top1_translation_error_m?: { median?: number; p90?: number }; top1_rotation_error_deg?: { median?: number; p90?: number } };
const pct = (value?: number) => value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
const num = (value?: number, suffix = "") => value === undefined ? "—" : `${value.toFixed(2)}${suffix}`;

export function EvaluationPage() {
  const [pose, setPose] = useState<Pose | null>(null);
  const [benchmark, setBenchmark] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { Promise.all([api.evaluation(), api.technicianBenchmark()]).then(([data, tasks]) => { setPose((data.pose ?? null) as Pose | null); setBenchmark(tasks.summary); }).catch((reason: Error) => setError(reason.message)); }, []);
  return <><header className="page-heading"><span className="eyebrow">Evidence Lab</span><h1>How reliable is this memory?</h1><p>Held-out Office sequences test whether visually similar frames lead back to the same physical place. Coverage is reported separately so an easy test set cannot hide missing reference views.</p></header>{error && <p className="error">{error}</p>}{!pose ? <p>Loading evaluation…</p> : <>
    <section className={styles.metricGrid} aria-label="Headline metrics"><article className={`panel ${styles.metric}`}><span>Held-out queries</span><strong>{pose.query_count?.toLocaleString() ?? "—"}</strong></article><article className={`panel ${styles.metric}`}><span>Strict coverage</span><strong>{pct(pose.strict?.coverage)}</strong></article><article className={`panel ${styles.metric}`}><span>Strict hit@1</span><strong>{pct(pose.strict?.hit_at_1?.all_query_rate)}</strong></article><article className={`panel ${styles.metric}`}><span>Strict hit@5</span><strong>{pct(pose.strict?.hit_at_5?.all_query_rate)}</strong></article></section>
    <section className={styles.section}><h2>Two definitions of “same place”</h2><table className={styles.table}><thead><tr><th>Protocol</th><th>Coverage</th><th>Hit@1</th><th>Hit@5</th><th>Hit@10</th></tr></thead><tbody><tr><td>Strict: within 0.25 m and 30°</td><td>{pct(pose.strict?.coverage)}</td><td>{pct(pose.strict?.hit_at_1?.all_query_rate)}</td><td>{pct(pose.strict?.hit_at_5?.all_query_rate)}</td><td>{pct(pose.strict?.hit_at_10?.all_query_rate)}</td></tr><tr><td>Relaxed: within 0.50 m and 30°</td><td>{pct(pose.relaxed?.coverage)}</td><td>{pct(pose.relaxed?.hit_at_1?.all_query_rate)}</td><td>{pct(pose.relaxed?.hit_at_5?.all_query_rate)}</td><td>{pct(pose.relaxed?.hit_at_10?.all_query_rate)}</td></tr></tbody></table></section>
    <section className={styles.section}><h2>Top result pose error</h2><div className={styles.metricGrid}><article className={`panel ${styles.metric}`}><span>Median translation</span><strong>{num(pose.top1_translation_error_m?.median, " m")}</strong></article><article className={`panel ${styles.metric}`}><span>90th percentile translation</span><strong>{num(pose.top1_translation_error_m?.p90, " m")}</strong></article><article className={`panel ${styles.metric}`}><span>Median rotation</span><strong>{num(pose.top1_rotation_error_deg?.median, "°")}</strong></article><article className={`panel ${styles.metric}`}><span>90th percentile rotation</span><strong>{num(pose.top1_rotation_error_deg?.p90, "°")}</strong></article></div></section>
    <section className={styles.section}><h2>By held-out sequence</h2><table className={styles.table}><thead><tr><th>Sequence</th><th>Strict coverage</th><th>Hit@1</th><th>Hit@5</th></tr></thead><tbody>{Object.entries(pose.per_sequence ?? {}).map(([sequence, row]) => <tr key={sequence}><td>{sequence}</td><td>{pct(row.strict_coverage)}</td><td>{pct(row.strict_hit_at_1_all_queries)}</td><td>{pct(row.strict_hit_at_5_all_queries)}</td></tr>)}</tbody></table></section>
    <p className={styles.section}><Link to="/lab/failures">Inspect individual successes and failures →</Link></p></>}</>;
}
