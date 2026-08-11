import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { ChangeCase, ChangeFocusBox, ChangePair, Phase6aShowcase } from "../types";
import styles from "./pages.module.css";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export function ChangesPage() {
  const [data, setData] = useState<Phase6aShowcase | null>(null);
  const [selectedPairId, setSelectedPairId] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let ignore = false;
    async function load() {
      try {
        const result = await api.phase6a();
        if (!ignore) {
          setData(result);
          setSelectedPairId(result.cases[0]?.pair_id ?? "");
        }
      } catch (reason) {
        if (!ignore) setError(reason instanceof Error ? reason.message : "Could not load Phase 6A.");
      }
    }
    void load();
    return () => { ignore = true; };
  }, []);

  const selectedCase = useMemo(
    () => data?.cases.find((item) => item.pair_id === selectedPairId) ?? null,
    [data, selectedPairId],
  );
  const pair = useMemo(
    () => data?.pairs.find((item) => item.pair_id === selectedPairId) ?? null,
    [data, selectedPairId],
  );

  if (error) return <><header className="page-heading"><span className="eyebrow">State-change memory</span><h1>What changed in the office?</h1></header><p className="error">{error}</p></>;
  if (!data) return <p>Loading the office comparisons...</p>;

  return <>
    <header className={`page-heading ${styles.changeHero}`}>
      <span className="eyebrow">Phase 6A · Repeated office visits</span>
      <h1>What changed since the last visit?</h1>
      <p>Choose two consecutive scans. The page shows the clearest RGB evidence first, then checks whether the 3D reconstruction tells the same story.</p>
    </header>

    {data.cases.length === 0 ? <p className="error">No curated consecutive-visit cases are available.</p> : <>
      <section className={styles.visitChooser} aria-label="Choose consecutive office visits">
        <span>Compare</span>
        <div className={styles.pairTabs}>
          {data.cases.map((item) => <button
            className={item.pair_id === selectedPairId ? styles.selectedTab : ""}
            key={item.pair_id}
            onClick={() => setSelectedPairId(item.pair_id)}
          >Visit {item.earlier_observation} → Visit {item.current_observation}</button>)}
        </div>
      </section>

      {selectedCase && pair && <ChangeStory key={selectedCase.pair_id} item={selectedCase} pair={pair} data={data} />}
    </>}
  </>;
}

function ChangeStory({ item, pair, data }: { item: ChangeCase; pair: ChangePair; data: Phase6aShowcase }) {
  const primary = pair.changed_fraction[data.method.primary_threshold_m.toFixed(3)];
  return <>
    <section className={`panel ${styles.changeAnswer}`}>
      <div>
        <span className={`${styles.outcomeBadge} ${styles[item.outcome]}`}>{item.outcome_label}</span>
        <h2>{item.headline}</h2>
        <p>{item.explanation}</p>
      </div>
      <div className={styles.confidenceNote}><strong>{item.confidence} confidence</strong><span>Curated visual interpretation</span></div>
    </section>

    <section className={styles.section}>
      <div className={styles.resultHeader}>
        <div><span className="eyebrow">Visible evidence</span><h2>Look at the highlighted area</h2></div>
        <span>Orange boxes are human-curated; no detector was used</span>
      </div>
      <div className={styles.focusCompare}>
        <FocusImage title={`Earlier · Visit ${item.earlier_observation}`} frame={item.earlier_frame} src={item.earlier_image_url} box={item.earlier_box} label="Area before" />
        <div className={styles.changeArrow} aria-hidden="true">→</div>
        <FocusImage title={`Later · Visit ${item.current_observation}`} frame={item.current_frame} src={item.current_image_url} box={item.current_box} label="Area after" />
      </div>
      <aside className={styles.limitNote}><strong>What we can safely say</strong><span>{item.limitation}</span></aside>
    </section>

    <section className={styles.section}>
      <div className={styles.resultHeader}>
        <div><span className="eyebrow">3D check</span><h2>Does the 3D scan support this movement?</h2></div>
      </div>
      <div className={`panel ${styles.geometryStory}`}>
        <img src={item.geometry_url} alt={`Focused 3D difference for visits ${item.earlier_observation} and ${item.current_observation}`} />
        <div>
          <h3>Geometry supports a changed region—not an object identity</h3>
          <p>{item.geometry_note}</p>
          <p className={styles.plainDefinition}><strong>In plain English:</strong> the scanner found surfaces here in one visit that did not line up with surfaces in the other visit. That can support the RGB observation, but it cannot name or track the object yet.</p>
        </div>
      </div>
    </section>

    <details className={`panel ${styles.diagnostics}`}>
      <summary>How did the comparison reach this result?</summary>
      <div className={styles.diagnosticsBody}>
        <ol className={styles.plainSteps}>
          <li><strong>Record the office twice.</strong><span>Each visit contains colour images, depth measurements, and camera movement.</span></li>
          <li><strong>Build a 3D copy of each visit.</strong><span>Depth measurements are combined into a room-shaped collection of surfaces.</span></li>
          <li><strong>Place both copies in the same coordinate system.</strong><span>This lets us compare the same physical parts of the room instead of comparing image pixels directly.</span></li>
          <li><strong>Find surfaces that no longer match.</strong><span>A surface counts as different when no surface lies within {(data.method.primary_threshold_m * 100).toFixed(0)} cm in the other visit.</span></li>
          <li><strong>Group nearby differences.</strong><span>Neighbouring changed surface cells become one candidate region for inspection.</span></li>
        </ol>

        <div className={styles.simpleStats}>
          <div><strong>{pair.current_only_candidate_count}</strong><span>separate regions seen only in the later scan</span></div>
          <div><strong>{pair.earlier_only_candidate_count}</strong><span>separate regions seen only in the earlier scan</span></div>
          {primary && <div><strong>{percent(Math.max(primary.current_only, primary.earlier_only))}</strong><span>largest unmatched surface fraction in this comparison</span></div>}
        </div>

        <p className={styles.diagnosticWarning}>These counts are deliberately not called “changes.” They also include occlusion, incomplete scanning, and reconstruction errors.</p>
        <div className={styles.compareGrid}>
          <EvidenceFigure title="Surfaces seen only later" caption="The numbered coloured regions are the largest areas selected for closer inspection." src={pair.current_only_projection_url} />
          <EvidenceFigure title="Surfaces seen only earlier" caption="A missing counterpart may mean removal, movement, poor coverage, or reconstruction error." src={pair.earlier_only_projection_url} />
        </div>
        <aside className={styles.claimBoundary}><strong>Research boundary</strong><span>{data.claim_boundary} The visit numbers describe order, not calendar dates.</span></aside>
      </div>
    </details>
  </>;
}

function FocusImage({ title, frame, src, box, label }: { title: string; frame: number; src: string; box: ChangeFocusBox; label: string }) {
  return <figure className={`panel ${styles.focusFigure}`}>
    <a href={src} target="_blank" rel="noreferrer" className={styles.focusImageWrap}>
      <img src={src} alt={`${title}, frame ${frame}`} />
      <span className={styles.focusBox} style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.width * 100}%`, height: `${box.height * 100}%` }}><em>{label}</em></span>
    </a>
    <figcaption><strong>{title}</strong><span>Frame {String(frame).padStart(6, "0")}</span></figcaption>
  </figure>;
}

function EvidenceFigure({ title, caption, src }: { title: string; caption: string; src: string }) {
  return <figure className={`panel ${styles.changeFigure}`}><a href={src} target="_blank" rel="noreferrer"><img src={src} alt={title} /></a><figcaption><strong>{title}</strong><span>{caption}</span></figcaption></figure>;
}
