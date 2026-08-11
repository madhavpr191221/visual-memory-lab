import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { Phase6aShowcase } from "../types";
import styles from "./pages.module.css";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export function ChangesPage() {
  const [data, setData] = useState<Phase6aShowcase | null>(null);
  const [selectedPairId, setSelectedPairId] = useState("");
  const [selectedObservation, setSelectedObservation] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    let ignore = false;
    async function load() {
      try {
        const result = await api.phase6a();
        if (!ignore) {
          setData(result);
          setSelectedPairId(result.pairs.find((pair) => pair.consecutive)?.pair_id ?? result.pairs[0]?.pair_id ?? "");
        }
      } catch (reason) {
        if (!ignore) setError(reason instanceof Error ? reason.message : "Could not load Phase 6A.");
      }
    }
    void load();
    return () => { ignore = true; };
  }, []);

  const pair = useMemo(
    () => data?.pairs.find((item) => item.pair_id === selectedPairId) ?? null,
    [data, selectedPairId],
  );
  const observation = data?.observations.find((item) => item.logical_order === selectedObservation) ?? null;
  const observationByIndex = useMemo(
    () => new Map(data?.observations.map((item) => [item.logical_order, item]) ?? []),
    [data],
  );

  if (error) return <><header className="page-heading"><span className="eyebrow">State-change memory</span><h1>What changed in the office?</h1></header><p className="error">{error}</p></>;
  if (!data) return <p>Loading the Phase 6A evidence...</p>;

  const earlier = pair ? observationByIndex.get(pair.earlier_observation) : undefined;
  const current = pair ? observationByIndex.get(pair.current_observation) : undefined;
  const primary = pair?.changed_fraction[data.method.primary_threshold_m.toFixed(3)];

  return <>
    <header className="page-heading">
      <span className="eyebrow">Phase 6A | Controlled 3D change</span>
      <h1>What changed between office scans?</h1>
      <p>This page follows the evidence from two real RGB-D scans to 3D difference clusters. It shows what the baseline can see, where it fragments, and what the VLM could or could not support.</p>
    </header>

    <aside className={styles.claimBoundary}><strong>Important boundary</strong><span>{data.claim_boundary} The observation numbers are logical order, not calendar dates.</span></aside>

    <section className={styles.metricGrid} aria-label="Phase 6A metrics">
      <Metric label="Office observations" value={data.metrics.observation_count} />
      <Metric label="Viewable RGB frames" value={data.metrics.rgb_sample_count} />
      <Metric label="Mesh comparisons" value={data.metrics.pair_count} />
      <Metric label="Raw geometric clusters" value={data.metrics.geometric_candidate_count} />
      <Metric label="VLM-reviewed candidates" value={data.metrics.reviewed_candidate_count} />
      <Metric label="Accepted pseudo-reference" value={data.metrics.accepted_pseudo_reference_count} />
    </section>

    <section className={styles.section}>
      <h2>From scan to evidence</h2>
      <div className={styles.processFlow}>
        {["Earlier RGB-D scan", "Aligned 3D reconstruction", "Nearest-surface comparison", "Changed voxel clusters", "RGB + 3D review"].map((step, index) => <div key={step}><span>{index + 1}</span><strong>{step}</strong></div>)}
      </div>
      <div className={`panel ${styles.formula}`}>
        <code>d(q, P) = min ||q - p||</code>
        <p>For every point <em>q</em> in one reconstruction, find the closest surface point <em>p</em> in the other. A residual above {(data.method.primary_threshold_m * 100).toFixed(0)} cm becomes a geometric change candidate.</p>
      </div>
    </section>

    <section className={styles.section}>
      <div className={styles.resultHeader}><div><span className="eyebrow">The raw observations</span><h2>See the office scans</h2></div><span>{observation?.frame_count ?? 0} sampled frames in this observation</span></div>
      <div className={styles.observationTabs}>{data.observations.map((item) => <button className={item.logical_order === selectedObservation ? styles.selectedTab : ""} key={item.observation_id} onClick={() => setSelectedObservation(item.logical_order)}>Observation {item.logical_order}</button>)}</div>
      {observation && <><img className={styles.contactSheet} src={observation.contact_sheet_url} alt={`Contact sheet for observation ${observation.logical_order}`} /><div className={styles.frameGrid}>{observation.frames.map((frame) => <a href={frame.image_url} target="_blank" rel="noreferrer" key={frame.message_index}><img loading="lazy" src={frame.image_url} alt={`Observation ${observation.logical_order}, frame ${frame.message_index}`} /><span>Frame {String(frame.message_index).padStart(6, "0")}</span></a>)}</div></>}
    </section>

    <section className={styles.section}>
      <div className={styles.resultHeader}><div><span className="eyebrow">Pairwise comparison</span><h2>Choose two logical visits</h2></div><span>Consecutive pairs are marked</span></div>
      <div className={styles.pairTabs}>{data.pairs.map((item) => <button className={item.pair_id === selectedPairId ? styles.selectedTab : ""} key={item.pair_id} onClick={() => setSelectedPairId(item.pair_id)}>{item.pair_id}{item.consecutive ? " | consecutive" : ""}</button>)}</div>
      {pair && earlier && current && <>
        <div className={styles.compareGrid}>
          <EvidenceFigure title={`Earlier observation ${pair.earlier_observation}`} caption="Eight representative RGB views used during review." src={earlier.vlm_contact_sheet_url} />
          <EvidenceFigure title={`Current observation ${pair.current_observation}`} caption="The same room in the later logical observation." src={current.vlm_contact_sheet_url} />
        </div>
        <div className={styles.compareGrid}>
          <EvidenceFigure title="Current-only geometry" caption={`${pair.current_only_candidate_count} clusters exceeded the 5 cm baseline.`} src={pair.current_only_projection_url} />
          <EvidenceFigure title="Earlier-only geometry" caption={`${pair.earlier_only_candidate_count} clusters exceeded the 5 cm baseline.`} src={pair.earlier_only_projection_url} />
        </div>
        {primary && <p className={styles.pairSummary}>At the 5 cm threshold, {percent(primary.current_only)} of current voxels and {percent(primary.earlier_only)} of earlier voxels lacked a nearby counterpart. This includes physical differences and reconstruction artifacts.</p>}
      </>}
    </section>

    {pair && <section className={styles.section}>
      <div className={styles.resultHeader}><div><span className="eyebrow">Structured review</span><h2>What did the VLM support?</h2></div><span>{pair.reviewed_candidates.length} largest candidates reviewed</span></div>
      <div className={styles.verdictLegend}><span>Supported: {data.metrics.verdict_counts.supported}</span><span>Uncertain: {data.metrics.verdict_counts.uncertain}</span><span>Unsupported: {data.metrics.verdict_counts.unsupported}</span></div>
      <div className={styles.reviewGrid}>{pair.reviewed_candidates.map((candidate) => <article className={`panel ${styles.reviewCard}`} key={candidate.candidate_id}><div className={styles.reviewTop}><span className={`${styles.verdict} ${styles[candidate.verdict]}`}>{candidate.verdict}</span><span>{candidate.confidence} confidence</span></div><code>{candidate.candidate_id}</code><h3>{candidate.interpretation.replaceAll("_", " ")}</h3><p>{candidate.description}</p>{candidate.limitations.length > 0 && <small>{candidate.limitations.join(" ")}</small>}</article>)}</div>
    </section>}
  </>;
}

function Metric({ label, value }: { label: string; value: number }) {
  return <article className={`panel ${styles.metric}`}><span>{label}</span><strong>{value.toLocaleString()}</strong></article>;
}

function EvidenceFigure({ title, caption, src }: { title: string; caption: string; src: string }) {
  return <figure className={`panel ${styles.changeFigure}`}><a href={src} target="_blank" rel="noreferrer"><img src={src} alt={title} /></a><figcaption><strong>{title}</strong><span>{caption}</span></figcaption></figure>;
}
