import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { AssociationPair, ObjectClass, Phase613Showcase } from "../types";
import styles from "./pages.module.css";

const label = (value: string) => value.replace("_", " ");
const score = (value: number) => value.toFixed(2);

export function AssociationPage() {
  const [data, setData] = useState<Phase613Showcase | null>(null);
  const [objectClass, setObjectClass] = useState<ObjectClass | "all">("all");
  const [pairId, setPairId] = useState("");
  const [error, setError] = useState("");
  useEffect(() => { let ignore = false; api.associations().then((value) => { if (!ignore) { setData(value); setPairId(value.pairs[0]?.pair_id ?? ""); } }).catch((reason: Error) => { if (!ignore) setError(reason.message); }); return () => { ignore = true; }; }, []);
  const pairs = useMemo(() => data?.pairs.filter((item) => objectClass === "all" || item.object_class === objectClass) ?? [], [data, objectClass]);
  const selected = pairs.find((item) => item.pair_id === pairId) ?? pairs[0] ?? null;
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p>Loading candidate associations...</p>;
  return <>
    <header className={`page-heading ${styles.objectHero}`}><span className="eyebrow">Phase 6.1.3 · candidate association</span><h1>Which detections may be the same object?</h1><p>This page ranks possible matches between visits. It does not assign a permanent identity. A movement label means possible movement only.</p></header>
    <aside className={styles.objectBoundary}><strong>Evidence boundary</strong><span>{data.claim_boundary}</span></aside>
    <section className={`panel ${styles.objectControls}`}><label>Object<select value={objectClass} onChange={(event) => { setObjectClass(event.target.value as ObjectClass | "all"); setPairId(""); }}><option value="all">All classes</option>{data.classes.map((item) => <option value={item} key={item}>{label(item)}</option>)}</select></label><label>Candidate pair<select value={selected?.pair_id ?? ""} onChange={(event) => setPairId(event.target.value)}>{pairs.slice(0, 300).map((item) => <option value={item.pair_id} key={item.pair_id}>Visit {item.earlier_observation} → {item.later_observation} · {label(item.object_class)} · {item.association_status}</option>)}</select></label></section>
    {selected ? <PairCard pair={selected} /> : <p>No candidate pairs are available.</p>}
    <section className={styles.objectMetrics} aria-label="Association summary"><Metric label="Candidate pairs" value={data.metrics.pair_count} /><Metric label="Detections" value={data.metrics.detection_count} /><Metric label="Shown" value={pairs.length} /></section>
  </>;
}

function PairCard({ pair }: { pair: AssociationPair }) { return <section className={`panel ${styles.rgbdComparison}`}><div className={styles.resultHeader}><div><span className="eyebrow">Candidate only</span><h2>{label(pair.object_class)} · Visit {pair.earlier_observation} → Visit {pair.later_observation}</h2></div><span className={`${styles.auditBadge} ${styles[pair.association_status]}`}>{pair.association_status}</span></div><div className={styles.rgbdImages}><Side item={pair.earlier} title="Earlier candidate" /><div className={styles.changeArrow}>→</div><Side item={pair.later} title="Later candidate" /></div><div className={styles.associationScores}><strong>Why this pair was ranked</strong><span>Appearance {score((pair.appearance_similarity + 1) / 2)} · shape {score(pair.shape_score)} · evidence {score(pair.evidence_score)} · position {score(pair.position_score)}</span><span>3D centroid distance: {pair.centroid_distance_m == null ? "unavailable" : `${pair.centroid_distance_m.toFixed(2)} m`}</span><strong>{pair.movement_status === "possible_movement" ? "Possible movement; identity still needs review." : "Movement not established."}</strong>{pair.vlm_audit && <p><strong>VLM pseudo-audit: {pair.vlm_audit.verdict}</strong> · {pair.vlm_audit.explanation}</p>}</div></section>; }
function Side({ item, title }: { item: AssociationPair["earlier"]; title: string }) { return <article className={styles.rgbdCard}><div className={styles.rgbdImage}><img src={item.image_url} alt={title} /><img className={styles.objectMask} src={item.mask_url} alt="" aria-hidden="true" /><span className={styles.associationBox} style={{ left: `${item.box_normalized[0] * 100}%`, top: `${item.box_normalized[1] * 100}%`, width: `${(item.box_normalized[2] - item.box_normalized[0]) * 100}%`, height: `${(item.box_normalized[3] - item.box_normalized[1]) * 100}%` }} /></div><h3>{title}</h3><p>Detector {item.score.toFixed(3)} · mask {item.mask_score.toFixed(3)}</p><p>{item.detection_id}</p></article>; }
function Metric({ label, value }: { label: string; value: number }) { return <div className={styles.objectMetric}><span>{label}</span><strong>{value.toLocaleString()}</strong></div>; }
