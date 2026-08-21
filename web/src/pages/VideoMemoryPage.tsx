import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { VideoCatalogResponse, VideoFollowUp, VideoGroundedAnswer, VideoMemoryResponse, VideoMemoryWindow, VideoObjectEvidence, VideoSummary } from "../types";
import styles from "./pages.module.css";

type Mode = "find" | "summarize";
type FindingStatus = "confirmed" | "unclear" | "needs_manual_review" | "rejected";

type SelectedEvidence = {
  videoId: string;
  query?: string;
  start: number;
  end: number;
  contextStart?: number;
  contextEnd?: number;
  label: string;
  contextActions?: string[];
  recordedAction?: VideoMemoryWindow["recorded_action"];
  resultLimitations?: string[];
  annotationStart?: number;
  annotationEnd?: number;
  refinementConfidence?: number;
  intervalSource?: VideoMemoryWindow["interval_source"];
  frameTimestamps?: number[];
  ids: string[];
};

const EXAMPLES = [
  "When did the person open the door?",
  "When did the person sit down?",
  "What happened before the person picked up the bag?",
];

function selectionFromWindow(item: VideoMemoryWindow, query: string): SelectedEvidence {
  const actionStart = item.action_start_s ?? item.recorded_action?.start_s ?? item.start_s;
  const actionEnd = item.action_end_s ?? item.recorded_action?.end_s ?? item.end_s;
  return { videoId: item.video_id, query, start: actionStart, end: actionEnd, contextStart: item.context_start_s ?? item.start_s, contextEnd: item.context_end_s ?? item.end_s, label: item.primary_action || "Relevant event", contextActions: item.context_actions, recordedAction: item.recorded_action, resultLimitations: item.result_limitations, annotationStart: item.annotation_start_s, annotationEnd: item.annotation_end_s, refinementConfidence: item.refinement_confidence, intervalSource: item.interval_source, frameTimestamps: item.frame_timestamps_s, ids: item.evidence_window_ids?.length ? item.evidence_window_ids : [item.window_id] };
}

export function VideoMemoryPage() {
  const [mode, setMode] = useState<Mode>("find");
  const [query, setQuery] = useState("");
  const [catalog, setCatalog] = useState<VideoCatalogResponse | null>(null);
  const [videoId, setVideoId] = useState("");
  const [result, setResult] = useState<VideoMemoryResponse | null>(null);
  const [summary, setSummary] = useState<VideoSummary | null>(null);
  const [selected, setSelected] = useState<SelectedEvidence | null>(null);
  const [question, setQuestion] = useState("");
  const [followUp, setFollowUp] = useState<VideoFollowUp | null>(null);
  const [synthesis, setSynthesis] = useState<VideoGroundedAnswer | null>(null);
  const [objectEvidence, setObjectEvidence] = useState<VideoObjectEvidence | null>(null);
  const [objectEvidenceLoading, setObjectEvidenceLoading] = useState(false);
  const [synthesisLoading, setSynthesisLoading] = useState(false);
  const [status, setStatus] = useState<FindingStatus>("unclear");
  const [note, setNote] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api.videoCatalog()
      .then(setCatalog)
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Could not load recordings."))
      .finally(() => setCatalogLoading(false));
  }, []);

  function resetEvidence() {
    setResult(null);
    setSummary(null);
    setSelected(null);
    setFollowUp(null);
    setSynthesis(null);
    setObjectEvidence(null);
    setObjectEvidenceLoading(false);
    setMessage("");
  }

  function changeVideo(value: string) {
    setVideoId(value);
    resetEvidence();
    setError("");
  }

  async function search(event: FormEvent) {
    event.preventDefault();
    if (!videoId || !query.trim()) return;
    setLoading(true);
    setError("");
    resetEvidence();
    try {
      const response = await api.videoMemory(query, videoId);
      setResult(response);
      if (response.results.length > 0) choose(selectionFromWindow(response.results[0], query));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Video search failed.");
    } finally {
      setLoading(false);
    }
  }

  async function summarize(event: FormEvent) {
    event.preventDefault();
    if (!videoId) return;
    setLoading(true);
    setError("");
    resetEvidence();
    try {
      setSummary(await api.summarizeVideo(videoId));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Recording summary failed.");
    } finally {
      setLoading(false);
    }
  }

  function choose(item: SelectedEvidence) {
    setSelected(item);
    setFollowUp(null);
    setMessage("");
    setError("");
    setSynthesisLoading(true);
    setObjectEvidenceLoading(true);
    const eventQuestion = item.query || query || "What happened in this event?";
    api.synthesizeVideo({ video_id: item.videoId, question: eventQuestion, event_label: item.label, start_s: item.start, end_s: item.end, evidence_window_ids: item.ids, mode: "detailed" })
      .then(setSynthesis)
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Event explanation failed."))
      .finally(() => setSynthesisLoading(false));
    api.videoObjectEvidence({ video_id: item.videoId, query: eventQuestion, event_label: item.label, start_s: item.start, end_s: item.end, frame_timestamps_s: item.frameTimestamps })
      .then(setObjectEvidence)
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Object inspection was unavailable."))
      .finally(() => setObjectEvidenceLoading(false));
  }

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!selected || !question.trim()) return;
    setLoading(true);
    setError("");
    try {
      setFollowUp(await api.videoFollowUp(selected.videoId, question, selected.start, selected.end));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Follow-up failed.");
    } finally {
      setLoading(false);
    }
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!selected || !followUp) return;
    setLoading(true);
    setError("");
    try {
      await api.createVideoFinding({
        video_id: selected.videoId,
        question: followUp.question,
        start_s: selected.start,
        end_s: selected.end,
        answer: followUp.answer,
        evidence_window_ids: followUp.evidence_window_ids,
        status,
        note,
        limitations: followUp.limitations,
        source: followUp.source,
      });
      setMessage("Finding saved. You can reopen it from Saved findings.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not save finding.");
    } finally {
      setLoading(false);
    }
  }

  const selectedRecording = catalog?.videos.find((item) => item.video_id === videoId);

  return <>
    <section className={styles.hero}>
      <div>
        <p className="eyebrow">Video memory</p>
        <h1>Find evidence in a recording.</h1>
        <p>Choose a recording, ask what you want to know, and review the exact moment before saving a finding.</p>
      </div>
      <aside className={styles.heroNote}>
        <strong>How this works</strong>
        <p>You choose the recording first. The system finds a moment, shows playable evidence, and explains what can safely be said.</p>
      </aside>
    </section>

    <section className={`panel ${styles.searchPanel}`}>
      <label className={styles.selectField}>
        <span>Recording</span>
        <select aria-label="Choose a recording" value={videoId} onChange={(event) => changeVideo(event.target.value)} disabled={catalogLoading}>
          <option value="">{catalogLoading ? "Loading recordings…" : "Choose a recording first"}</option>
          {catalog?.videos.map((video) => <option key={video.video_id} value={video.video_id}>{video.video_id} · {video.duration_s.toFixed(1)} seconds</option>)}
        </select>
      </label>
      {selectedRecording && <div className={styles.fieldHint}><p><strong>Recording summary:</strong> {selectedRecording.description || "No dataset summary was supplied."}</p><small>Action labels, objects, and exact timestamps are hidden during retrieval so the question tests the memory system rather than the annotation list.</small></div>}
      <div className={styles.modeRow}>
        <div className={styles.segments} role="tablist" aria-label="Video task">
          <button type="button" className={mode === "find" ? styles.active : ""} onClick={() => setMode("find")}>Find an event</button>
          <button type="button" className={mode === "summarize" ? styles.active : ""} onClick={() => setMode("summarize")}>Review the timeline</button>
        </div>
      </div>
      {mode === "find" ? <form onSubmit={search}>
        <div className={styles.question}>
          <textarea aria-label="Video memory question" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={videoId ? "What would you like to find in this recording?" : "Choose a recording before asking a question."} disabled={!videoId} />
          <button className={styles.primary} disabled={loading || !videoId || !query.trim()}>{loading ? "Finding evidence…" : "Find evidence"}</button>
        </div>
        <div className={styles.examples}>{EXAMPLES.map((example) => <button type="button" key={example} disabled={!videoId} onClick={() => setQuery(example)}>{example}</button>)}</div>
      </form> : <form onSubmit={summarize}>
        <p className={styles.fieldHint}>Review the actions that the recording explicitly marks with time intervals.</p>
        <button className={styles.primary} disabled={loading || !videoId}>{loading ? "Preparing timeline…" : "Review timeline"}</button>
      </form>}
      {!videoId && !catalogLoading && <p className={styles.emptyState}>Choose a recording to begin. Your question will be answered only against that recording.</p>}
      {error && <p className="error" role="alert">{error}</p>}
    </section>

    {mode === "find" && result && <SearchResults result={result} query={query} selected={selected} choose={choose} objectEvidence={objectEvidence} synthesis={synthesis} synthesisLoading={synthesisLoading} />}
    {mode === "summarize" && summary && <TimelineReview summary={summary} choose={choose} />}
    {selected && <EvidenceReview selected={selected} question={question} setQuestion={setQuestion} followUp={followUp} objectEvidence={objectEvidence} objectEvidenceLoading={objectEvidenceLoading} loading={loading} error={error} message={message} status={status} setStatus={setStatus} note={note} setNote={setNote} ask={ask} save={save} />}
  </>;
}

function SearchResults({ result, query, selected, choose, objectEvidence, synthesis, synthesisLoading }: { result: VideoMemoryResponse; query: string; selected: SelectedEvidence | null; choose: (item: SelectedEvidence) => void; objectEvidence: VideoObjectEvidence | null; synthesis: VideoGroundedAnswer | null; synthesisLoading: boolean }) {
  return <section className={styles.results} aria-live="polite">
    <div className={`panel ${styles.answerStrip}`}>
       <div><span className="status strong">Candidate events</span><h2>{result.results.length} possible event{result.results.length === 1 ? "" : "s"}</h2><p>Each card is one distinct moment to review.</p></div>
    </div>
    {synthesisLoading && <div className={`panel ${styles.emptyState}`}><p>Preparing a grounded explanation from the strongest evidence…</p></div>}
    {result.results.length === 0 ? <div className={`panel ${styles.emptyState}`}><h2>No supported event found</h2><p>{result.message || "This recording does not contain a matching event in the available evidence. Try a broader question or review the timeline."}</p></div> : <div className={styles.evidenceGrid}>{result.results.map((item) => {
      const label = item.primary_action || "Relevant event";
      const isSelected = selected?.videoId === item.video_id && Math.abs(selected.start - (item.action_start_s ?? item.recorded_action?.start_s ?? item.start_s)) < 0.01;
        const actionStart = item.action_start_s ?? item.recorded_action?.start_s ?? item.start_s;
        const actionEnd = item.action_end_s ?? item.recorded_action?.end_s ?? item.end_s;
        const contextStart = item.context_start_s ?? item.start_s;
        const contextEnd = item.context_end_s ?? item.end_s;
         const selection = selectionFromWindow(item, query);
        return <article className={`panel ${styles.evidenceCard} ${isSelected ? styles.selectedEvidence : ""}`} key={item.event_id ?? item.window_id}>
        {isSelected && objectEvidence ? <ObjectEvidencePanel evidence={objectEvidence} playback /> : item.video_url && <video controls preload="metadata" src={`${item.video_url}#t=${contextStart},${contextEnd}`} onClick={(event) => event.stopPropagation()} />}
        {item.frame_timestamps_s?.length ? <FrameStrip videoId={item.video_id} label={label} timestamps={item.frame_timestamps_s} /> : null}
        <div className={styles.evidenceBody}><div className={styles.evidenceTop}><span className={styles.rank}>{item.video_id}</span><span className={styles.score}>Action {actionStart.toFixed(1)}–{actionEnd.toFixed(1)} s</span></div><h3>{label}</h3><div className={styles.evidenceRows}><p><strong>Matched action interval</strong><span>{actionStart.toFixed(1)}–{actionEnd.toFixed(1)} s</span></p><p><strong>Context shown</strong><span>{contextStart.toFixed(1)}–{contextEnd.toFixed(1)} s</span></p><p><strong>Annotation note</strong><span>{item.recorded_action?.note || "Dataset annotation; not independent visual proof."}</span></p><p><strong>Visual review</strong><span>{isSelected && synthesis ? (synthesis.visible_evidence || synthesis.answer) : "Select this event to inspect sampled frames."}</span></p></div>{item.context_actions?.length ? <small><strong>Overlapping context:</strong> {item.context_actions.join(" · ")}</small> : null}{item.objects.length > 0 && <small>Associated objects: {item.objects.join(" · ")}</small>}<button type="button" className={styles.secondary} onClick={() => choose(selection)}>{isSelected ? "Refresh object evidence" : "Inspect event and objects"}</button></div>
      </article>;
    })}</div>}
  </section>;
}

function FrameStrip({ videoId, label, timestamps }: { videoId: string; label: string; timestamps: number[] }) {
  return <div className={styles.frameStrip} aria-label="Sampled evidence frames">{timestamps.map((timestamp) => <figure key={timestamp}><img src={`/api/video-memory/frame/${videoId}?timestamp_s=${timestamp}`} alt={`${label} at ${timestamp.toFixed(1)} seconds`} /><figcaption>{timestamp.toFixed(1)} s</figcaption></figure>)}</div>;
}

function TimelineReview({ summary, choose }: { summary: VideoSummary; choose: (item: SelectedEvidence) => void }) {
  return <section className={styles.results} aria-live="polite"><div className={`panel ${styles.answerStrip}`}><div><span className="status strong">Recording timeline</span><h2>{summary.video_id}</h2><p>{summary.overview || "The recording has official action annotations."}</p></div><div className={styles.timeNotice}>Timed actions are the events explicitly marked by the dataset.</div></div><div className={`panel ${styles.timeline}`}><video controls preload="metadata" src={summary.video_url} /><h2>Timed actions</h2><p className={styles.timelineNote}>Select an action to review it and ask a follow-up question.</p>{summary.events.map((item) => <button type="button" className={styles.timelineEvent} key={`${item.label}-${item.start_s}`} onClick={() => choose({ videoId: summary.video_id, start: item.start_s, end: item.end_s, label: item.label, ids: item.source_events?.map((event) => event.evidence_window_id) ?? [] })}><span>{item.start_s.toFixed(1)}–{item.end_s.toFixed(1)} s</span><strong>{item.label}</strong><small>Grouped from {item.source_events?.length ?? 1} source annotation(s).</small></button>)}</div></section>;
}

function EvidenceReview(props: { selected: SelectedEvidence; question: string; setQuestion: (value: string) => void; followUp: VideoFollowUp | null; objectEvidence: VideoObjectEvidence | null; objectEvidenceLoading: boolean; loading: boolean; error: string; message: string; status: FindingStatus; setStatus: (value: FindingStatus) => void; note: string; setNote: (value: string) => void; ask: (event: FormEvent) => void; save: (event: FormEvent) => void }) {
  const { selected } = props;
  const setProps = { status: props.setStatus, question: props.setQuestion };
  return <section className={`panel ${styles.followUp}`}><span className="eyebrow">Inspection report</span><h2>{selected.label}</h2><p className={styles.fieldHint}>Review the exact event, then check the visible objects before saving a finding.</p><video controls preload="metadata" src={`/api/video-memory/videos/${selected.videoId}#t=${Math.max(0, (selected.contextStart ?? selected.start) - 2)},${(selected.contextEnd ?? selected.end) + 2}`} /><p>Event interval: {selected.start.toFixed(1)}–{selected.end.toFixed(1)} seconds. The player includes surrounding context for review.</p>{props.objectEvidence ? <ObjectEvidencePanel evidence={props.objectEvidence} /> : <div className={styles.objectEvidence}><strong>{props.objectEvidenceLoading ? "Inspecting visible objects…" : "Object evidence unavailable"}</strong><p className={styles.fieldHint}>{props.objectEvidenceLoading ? "The detector and optional segmenter are reviewing sampled RGB frames. This can take a while the first time." : "No object report was returned. The event video and annotation evidence remain available."}</p></div>}{props.followUp && <form className={styles.analysis} onSubmit={props.save}><span className={`status ${props.followUp.supported ? "supported" : "uncertain"}`}>{props.followUp.supported ? "Evidence-supported" : "Not established"}</span><p>{props.followUp.answer}</p><small>{props.followUp.limitations[0]}</small><label>Status<select value={props.status} onChange={(event) => setProps.status(event.target.value as FindingStatus)}><option value="unclear">Unclear</option><option value="confirmed">Confirmed</option><option value="needs_manual_review">Needs manual review</option><option value="rejected">Rejected</option></select></label><textarea aria-label="Finding note" value={props.note} onChange={(event) => props.setNote(event.target.value)} placeholder="Optional note for the finding" /><button className={styles.secondary} disabled={props.loading}>Save finding</button>{props.message && <p className="success">{props.message}</p>}</form>}<form onSubmit={props.ask}><div className={styles.question}><textarea aria-label="Follow-up question" value={props.question} onChange={(event) => setProps.question(event.target.value)} placeholder="Ask about this selected event" /><button className={styles.primary} disabled={props.loading || !props.question.trim()}>{props.loading ? "Checking…" : "Ask"}</button></div></form>{props.error && <p className="error" role="alert">{props.error}</p>}</section>;
}

function ObjectEvidencePanel({ evidence, playback = false }: { evidence: VideoObjectEvidence; playback?: boolean }) {
  const [time, setTime] = useState(evidence.start_s);
  const nearest = evidence.frames.reduce<(typeof evidence.frames)[number] | null>((best, frame) => !best || Math.abs(frame.timestamp_s - time) < Math.abs(best.timestamp_s - time) ? frame : best, null);
  const inEvent = time >= evidence.start_s && time <= evidence.end_s;
  const contextStart = Math.max(0, evidence.start_s - 2);
  const contextEnd = evidence.end_s + 2;
  return <div className={styles.objectEvidence}><div className={styles.evidenceTop}><strong>Event playback with object evidence</strong><span className="status">{evidence.status === "detected" ? "Model review" : "Unavailable"}</span></div><div className={styles.overlayPlayer}><div className={styles.overlayStage}><video controls preload="metadata" src={`/api/video-memory/videos/${evidence.video_id}#t=${contextStart},${contextEnd}`} onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)} /><div className={styles.overlayLayer}>{inEvent && nearest?.detections.map((detection, index) => <span key={`${detection.track_id ?? detection.label}-${index}`} className={styles.videoObjectBox} style={{ left: `${detection.box_normalized[0] * 100}%`, top: `${detection.box_normalized[1] * 100}%`, width: `${(detection.box_normalized[2] - detection.box_normalized[0]) * 100}%`, height: `${(detection.box_normalized[3] - detection.box_normalized[1]) * 100}%` }}><b>{detection.label} · {detection.score.toFixed(2)}</b></span>)}</div></div><div className={styles.overlayMeta}><strong>Matched event: {evidence.start_s.toFixed(1)}–{evidence.end_s.toFixed(1)} s</strong><span>Context shown: {contextStart.toFixed(1)}–{contextEnd.toFixed(1)} s</span><span>{inEvent ? "Object overlay active" : "Context playback · event overlay paused"}</span></div></div>{evidence.objects.length > 0 && <div className={styles.objectSummary}>{evidence.objects.map((item) => <div key={item.label}><strong>{item.label}</strong><span>{item.status.replaceAll("_", " ")} · {item.frames_visible}/{item.frame_count} frames · score {item.max_score.toFixed(2)}</span></div>)}</div>}{evidence.limitations.map((item) => <small key={item}>{item}</small>)}</div>;
}

function LegacyObjectEvidencePanel({ evidence }: { evidence: VideoObjectEvidence }) {
  return <div className={styles.objectEvidence}><div className={styles.evidenceTop}><strong>Visible object evidence</strong><span className="status">{evidence.status === "detected" ? "Model review" : "Unavailable"}</span></div>{evidence.objects.length ? <div className={styles.objectSummary}>{evidence.objects.map((item) => <div key={item.label}><strong>{item.label}</strong><span>{item.status.replaceAll("_", " ")} · {item.frames_visible}/{item.frame_count} frames · score {item.max_score.toFixed(2)}</span></div>)}</div> : <p>No object detections were available for this event.</p>}{evidence.frames.length > 0 && <div className={styles.objectFrameGrid}>{evidence.frames.map((frame) => <figure key={frame.frame_id} className={styles.objectFrame}><div><img src={`/api/video-memory/frame/${evidence.video_id}?timestamp_s=${frame.timestamp_s}`} alt={`Object evidence at ${frame.timestamp_s.toFixed(1)} seconds`} /><svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">{frame.detections.map((detection, index) => <rect key={`${detection.label}-${index}`} x={detection.box_normalized[0]} y={detection.box_normalized[1]} width={detection.box_normalized[2] - detection.box_normalized[0]} height={detection.box_normalized[3] - detection.box_normalized[1]} />)}</svg></div><figcaption>{frame.timestamp_s.toFixed(1)} s · {frame.detections.length} detection{frame.detections.length === 1 ? "" : "s"}</figcaption></figure>)}</div>}{evidence.limitations.map((item) => <small key={item}>{item}</small>)}</div>;
}
