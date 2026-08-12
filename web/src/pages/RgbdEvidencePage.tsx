import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { ObjectClass, Phase612Showcase, RgbdComparison, RgbdEvidence } from "../types";
import styles from "./pages.module.css";

const label = (value: string) => value.replace("_", " ");
const metres = (value: number | null) => value == null ? "—" : `${value.toFixed(2)} m`;

export function RgbdEvidencePage() {
  const [data, setData] = useState<Phase612Showcase | null>(null);
  const [error, setError] = useState("");
  const [objectClass, setObjectClass] = useState<ObjectClass | "all">("all");
  const [pair, setPair] = useState("");

  useEffect(() => {
    let ignore = false;
    api.phase612().then((result) => { if (!ignore) { setData(result); setPair(result.comparisons[0]?.id ?? ""); } })
      .catch((reason: Error) => { if (!ignore) setError(reason.message); });
    return () => { ignore = true; };
  }, []);

  const comparisons = useMemo(() => data?.comparisons.filter((item) => objectClass === "all" || item.object_class === objectClass) ?? [], [data, objectClass]);
  const selected = comparisons.find((item) => item.id === pair) ?? comparisons[0] ?? null;
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p>Loading RGB-D object evidence...</p>;
  return <>
    <header className={`page-heading ${styles.objectHero}`}>
      <span className="eyebrow">Phase 6.1.2 · recorded RGB-D evidence</span>
      <h1>Compare the visible evidence between visits</h1>
      <p>Choose an object class and two logical visits. The page shows what the RGB masks and recorded point clouds support; it does not claim that two detections are the same object.</p>
    </header>
    <aside className={styles.objectBoundary}><strong>Evidence boundary</strong><span>{data.claim_boundary}</span></aside>
    <section className={`panel ${styles.objectControls}`}>
      <label>Object<select value={objectClass} onChange={(event) => { setObjectClass(event.target.value as ObjectClass | "all"); setPair(""); }}><option value="all">All classes</option>{data.classes.map((item) => <option value={item} key={item}>{label(item)}</option>)}</select></label>
      <label>Visit pair<select value={selected?.id ?? ""} onChange={(event) => setPair(event.target.value)}>{comparisons.map((item) => <option value={item.id} key={item.id}>Visit {item.earlier_observation} → Visit {item.later_observation} · {label(item.object_class)}</option>)}</select></label>
    </section>
    {selected ? <Comparison comparison={selected} /> : <p>No pair has usable point-cloud evidence for this class.</p>}
    <section className={styles.objectMetrics} aria-label="RGB-D summary"><Metric label="RGB frames" value={data.metrics.frame_count} /><Metric label="Detections" value={data.metrics.detection_count} /><Metric label="Evidence records" value={data.metrics.evidence_count} /><Metric label="With point-cloud evidence" value={data.metrics.nonempty_evidence_count} /></section>
  </>;
}

function Comparison({ comparison }: { comparison: RgbdComparison }) {
  return <section className={`panel ${styles.rgbdComparison}`}>
    <div className={styles.resultHeader}><div><span className="eyebrow">Visible evidence only</span><h2>{label(comparison.object_class)} · Visit {comparison.earlier_observation} → Visit {comparison.later_observation}</h2></div><span>Not an identity or movement claim</span></div>
    <div className={styles.rgbdImages}><EvidenceCard title={`Earlier · Visit ${comparison.earlier_observation}`} item={comparison.earlier} /><div className={styles.changeArrow}>→</div><EvidenceCard title={`Later · Visit ${comparison.later_observation}`} item={comparison.later} /></div>
    <div className={styles.rgbdConclusion}><strong>What this phase can say</strong><p>{comparison.interpretation}</p></div>
  </section>;
}

function EvidenceCard({ title, item }: { title: string; item: RgbdEvidence }) {
  return <article className={styles.rgbdCard}><div className={styles.rgbdImage}><img src={item.image_url} alt={`${title} RGB evidence`} /><img className={styles.objectMask} src={item.mask_url} alt="" aria-hidden="true" /></div><h3>{title}</h3><p>{item.point_count.toLocaleString()} visible point-cloud points</p><p>Room-frame centroid: {item.centroid_world_m.map(metres).join(" / ")}</p><ExtentSketch item={item} /><p>Visible extent: {metres(item.extent_world_m.minimum[0])} to {metres(item.extent_world_m.maximum[0])} on X</p></article>;
}

function ExtentSketch({ item }: { item: RgbdEvidence }) {
  const valid = item.centroid_world_m.every((value) => value != null);
  return <div className={styles.extentSketch}><span>3D extent sketch (not a mesh)</span><svg viewBox="0 0 200 100" role="img" aria-label="Approximate visible 3D extent"><line x1="20" y1="80" x2="180" y2="80" /><line x1="20" y1="80" x2="20" y2="15" /><rect x="48" y="30" width="104" height="40" /><circle cx="100" cy="50" r="4" className={valid ? styles.extentPoint : ""} /></svg></div>;
}

function Metric({ label, value }: { label: string; value: number }) { return <div className={styles.objectMetric}><span>{label}</span><strong>{value.toLocaleString()}</strong></div>; }
