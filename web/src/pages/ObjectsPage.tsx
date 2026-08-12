import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type {
  ObjectAuditStatus,
  ObjectClass,
  ObjectDetection,
  ObjectFrame,
  Phase6b1Showcase,
} from "../types";
import styles from "./pages.module.css";

type DisplayMode = "raw" | "boxes" | "masks" | "combined";
type ClassFilter = "all" | ObjectClass;
type VisitFilter = "all" | "0" | "1" | "2" | "3";
type StatusFilter = "all" | ObjectAuditStatus;

const classLabel = (value: ObjectClass) => value.replace("_", " ");
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export function ObjectsPage() {
  const [data, setData] = useState<Phase6b1Showcase | null>(null);
  const [error, setError] = useState("");
  const [visit, setVisit] = useState<VisitFilter>("all");
  const [objectClass, setObjectClass] = useState<ClassFilter>("all");
  const [auditStatus, setAuditStatus] = useState<StatusFilter>("all");
  const [minimumScore, setMinimumScore] = useState(0.25);
  const [displayMode, setDisplayMode] = useState<DisplayMode>("combined");
  const [selectedFrameId, setSelectedFrameId] = useState("");
  const [visibleCount, setVisibleCount] = useState(48);

  useEffect(() => {
    let ignore = false;
    async function load() {
      try {
        const result = await api.objects();
        if (!ignore) {
          setData(result);
          // Keep the permissive research threshold available, but begin at the
          // separately reported pseudo-audit slice used for failure inspection.
          setMinimumScore(Math.max(result.method.box_threshold, 0.35));
          setSelectedFrameId(result.frames[0]?.frame_id ?? "");
        }
      } catch (reason) {
        if (!ignore) setError(reason instanceof Error ? reason.message : "Could not load Phase 6B1.");
      }
    }
    void load();
    return () => { ignore = true; };
  }, []);

  const detectionsFor = (frame: ObjectFrame) => frame.detections.filter((item) =>
    item.score >= minimumScore
    && (objectClass === "all" || item.canonical_class === objectClass)
    && (auditStatus === "all" || item.audit_status === auditStatus),
  );

  const filteredFrames = useMemo(() => data?.frames.filter((frame) => {
    if (visit !== "all" && frame.observation !== Number(visit)) return false;
    const detections = detectionsFor(frame);
    if (objectClass !== "all" || auditStatus !== "all") return detections.length > 0;
    return frame.detections.some((item) => item.score >= minimumScore) || frame.detections.length === 0;
  }) ?? [], [data, visit, objectClass, auditStatus, minimumScore]);

  const selectedFrame = useMemo(
    () => filteredFrames.find((item) => item.frame_id === selectedFrameId) ?? filteredFrames[0] ?? null,
    [filteredFrames, selectedFrameId],
  );
  const selectedDetections = selectedFrame ? detectionsFor(selectedFrame) : [];

  useEffect(() => { setVisibleCount(48); }, [visit, objectClass, auditStatus, minimumScore]);

  if (error) return <><header className="page-heading"><span className="eyebrow">Phase 6B1</span><h1>Automatic object localization</h1></header><p className="error">{error}</p></>;
  if (!data) return <p>Loading model-generated object evidence...</p>;

  const detectorName = String(data.method.detector.model_id ?? "Grounding DINO");
  const segmenterName = String(data.method.segmenter.model_id ?? "SAM 2.1");
  const auditRate = data.audit?.high_confidence_pseudo_support_rate;
  const failureFrames = data.frames.filter((frame) =>
    frame.missed_visible_classes.length > 0
    || frame.detections.some((item) => item.audit_status === "unsupported" || item.audit_status === "uncertain"),
  );

  return <>
    <header className={`page-heading ${styles.objectHero}`}>
      <span className="eyebrow">Phase 6B1 · Frozen RGB baseline</span>
      <h1>What objects can the memory actually see?</h1>
      <p>Inspect genuine model predictions across four office visits. Grounding DINO draws the boxes; SAM 2.1 separates the object pixels. Nothing on this page is a hand-drawn presentation box.</p>
    </header>

    <aside className={styles.objectBoundary}><strong>Evidence boundary</strong><span>{data.claim_boundary}</span></aside>

    <section className={styles.objectMetrics} aria-label="Localization summary">
      <Metric label="Dense RGB keyframes" value={data.metrics.frame_count} />
      <Metric label="Object predictions" value={data.metrics.detection_count} />
      <Metric label="Frames with detections" value={data.metrics.frames_with_detections} />
      <Metric label="Chairs" value={data.metrics.class_counts.chair ?? 0} />
      <Metric label="Bins" value={data.metrics.class_counts.waste_bin ?? 0} />
      <Metric label="Boxes" value={data.metrics.class_counts.box ?? 0} />
      <Metric label="VLM pseudo-support" value={auditRate == null ? "Not audited" : percent(auditRate)} />
    </section>

    <section className={`panel ${styles.objectControls}`} aria-label="Filter object predictions">
      <label>Visit<select value={visit} onChange={(event) => setVisit(event.target.value as VisitFilter)}><option value="all">All visits</option>{[0, 1, 2, 3].map((item) => <option value={item} key={item}>Visit {item}</option>)}</select></label>
      <label>Object<select value={objectClass} onChange={(event) => setObjectClass(event.target.value as ClassFilter)}><option value="all">All target objects</option><option value="chair">Chairs</option><option value="waste_bin">Waste bins</option><option value="box">Boxes</option></select></label>
      <label>Audit<select value={auditStatus} onChange={(event) => setAuditStatus(event.target.value as StatusFilter)}><option value="all">All audit states</option><option value="supported">Supported</option><option value="uncertain">Uncertain</option><option value="unsupported">Unsupported</option><option value="unreviewed">Not VLM-audited</option></select></label>
      <label className={styles.scoreControl}>Minimum detector score <strong>{minimumScore.toFixed(2)}</strong><input aria-label="Minimum detector score" type="range" min={data.method.box_threshold} max="0.75" step="0.05" value={minimumScore} onChange={(event) => setMinimumScore(Number(event.target.value))} /></label>
    </section>

    {selectedFrame && <section className={styles.objectInspector}>
      <div className={styles.resultHeader}><div><span className="eyebrow">Selected evidence</span><h2>Visit {selectedFrame.observation}, frame {String(selectedFrame.message_index).padStart(6, "0")}</h2></div><span>{selectedDetections.length} visible prediction{selectedDetections.length === 1 ? "" : "s"} after filtering</span></div>
      <div className={styles.displayTabs}>{(["raw", "boxes", "masks", "combined"] as DisplayMode[]).map((mode) => <button key={mode} className={displayMode === mode ? styles.selectedTab : ""} onClick={() => setDisplayMode(mode)}>{mode}</button>)}</div>
      <div className={`panel ${styles.objectDetail}`}>
        <ObjectImage frame={selectedFrame} detections={selectedDetections} mode={displayMode} />
        <div className={styles.objectDetails}>
          {selectedDetections.length === 0 ? <p>No target-object prediction survives the current filters. That does not prove the scene is empty.</p> : selectedDetections.map((item) => <DetectionDetail key={item.detection_id} item={item} />)}
          {selectedFrame.missed_visible_classes.length > 0 && <p className={styles.missedWarning}><strong>Possible VLM-noted miss:</strong> {selectedFrame.missed_visible_classes.map(classLabel).join(", ")}</p>}
        </div>
      </div>
    </section>}

    <section className={styles.section}>
      <div className={styles.resultHeader}><div><span className="eyebrow">All keyframes</span><h2>Browse the automatic predictions</h2></div><span>{filteredFrames.length} frames match these filters</span></div>
      <div className={styles.objectGrid}>{filteredFrames.slice(0, visibleCount).map((frame) => <button className={`${styles.objectCard} ${selectedFrame?.frame_id === frame.frame_id ? styles.selectedObjectCard : ""}`} key={frame.frame_id} onClick={() => setSelectedFrameId(frame.frame_id)}><img src={frame.overlay_url} alt={`Object predictions for visit ${frame.observation}, frame ${frame.message_index}`} loading="lazy" /><span><strong>Visit {frame.observation} · frame {String(frame.message_index).padStart(6, "0")}</strong><small>{detectionsFor(frame).length} filtered · {frame.audit_status}</small></span></button>)}</div>
      {visibleCount < filteredFrames.length && <button className={styles.loadMore} onClick={() => setVisibleCount((value) => value + 48)}>Show 48 more frames</button>}
    </section>

    <section className={styles.section}>
      <div className={styles.resultHeader}><div><span className="eyebrow">Failure inspection</span><h2>Where should we distrust the baseline?</h2></div><span>{failureFrames.length} audited frames contain uncertainty, disagreement, or a possible miss</span></div>
      {data.audit ? <>
        <div className={styles.failureSummary}>
          <Metric label="Supported" value={data.audit.verdict_counts.supported ?? 0} />
          <Metric label="Uncertain" value={data.audit.verdict_counts.uncertain ?? 0} />
          <Metric label="Unsupported" value={data.audit.verdict_counts.unsupported ?? 0} />
          <Metric label="Possible missed classes" value={Object.values(data.audit.missed_visible_class_counts).reduce((sum, value) => sum + value, 0)} />
        </div>
        <p className={styles.auditBoundary}>{data.audit.claim_boundary}</p>
      </> : <p className={styles.auditBoundary}>The optional 48-frame VLM pseudo-audit has not been generated. Predictions remain available for direct inspection.</p>}
    </section>

    <details className={`panel ${styles.diagnostics}`}>
      <summary>Model and threshold details</summary>
      <div className={styles.diagnosticsBody}><p><strong>Detector:</strong> {detectorName}</p><p><strong>Segmenter:</strong> {segmenterName}</p><p><strong>Prompt:</strong> {data.method.prompt}</p><p><strong>Thresholds:</strong> box {data.method.box_threshold.toFixed(2)}, text {data.method.text_threshold.toFixed(2)}, duplicate suppression IoU {data.method.nms_iou.toFixed(2)}.</p><p>Detection identifies a category and rectangle. Segmentation estimates its pixels. Neither operation establishes that two visits contain the same physical object.</p></div>
    </details>
  </>;
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return <div className={styles.objectMetric}><span>{label}</span><strong>{value}</strong></div>;
}

function ObjectImage({ frame, detections, mode }: { frame: ObjectFrame; detections: ObjectDetection[]; mode: DisplayMode }) {
  return <a href={frame.image_url} target="_blank" rel="noreferrer" className={styles.objectImageStage}>
    <img src={frame.image_url} alt={`Raw office frame ${frame.message_index}`} />
    {(mode === "masks" || mode === "combined") && detections.map((item) => <img className={styles.objectMask} src={item.mask_url} alt="" aria-hidden="true" key={`${item.detection_id}-mask`} />)}
    {(mode === "boxes" || mode === "combined") && detections.map((item) => {
      const [x1, y1, x2, y2] = item.box_normalized;
      return <span className={`${styles.objectBox} ${styles[item.canonical_class]}`} key={`${item.detection_id}-box`} style={{ left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` }}><em>{classLabel(item.canonical_class)} {item.score.toFixed(2)}</em></span>;
    })}
  </a>;
}

function DetectionDetail({ item }: { item: ObjectDetection }) {
  return <article className={styles.detectionDetail}>
    <div><strong>{classLabel(item.canonical_class)}</strong><span className={`${styles.auditBadge} ${styles[item.audit_status]}`}>{item.audit_status}</span></div>
    <p>Detector score {item.score.toFixed(3)} · SAM mask score {item.mask_score.toFixed(3)} · mask covers {percent(item.mask_area_fraction)}</p>
    <small>Prompt phrase: “{item.phrase}”</small>
    {item.audit && <p>{item.audit.explanation} Mask: {item.audit.mask_quality}.</p>}
    {item.warnings.length > 0 && <p className={styles.missedWarning}>Automatic warning: {item.warnings.join(", ")}</p>}
  </article>;
}
