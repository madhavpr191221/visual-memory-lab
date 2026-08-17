import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";
import type { VideoCatalogResponse, VideoFollowUp, VideoMemoryResponse, VideoMemoryWindow, VideoSummary } from "../types";
import styles from "./pages.module.css";

type Mode = "find" | "summarize";
type FindingStatus = "confirmed" | "unclear" | "needs_manual_review" | "rejected";

export function VideoMemoryPage() {
  const [mode, setMode] = useState<Mode>("find");
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<VideoMemoryResponse | null>(null);
  const [catalog, setCatalog] = useState<VideoCatalogResponse | null>(null);
  const [videoId, setVideoId] = useState("");
  const [summary, setSummary] = useState<VideoSummary | null>(null);
  const [selected, setSelected] = useState<{ videoId: string; start: number; end: number; label: string; ids: string[] } | null>(null);
  const [question, setQuestion] = useState("");
  const [followUp, setFollowUp] = useState<VideoFollowUp | null>(null);
  const [status, setStatus] = useState<FindingStatus>("unclear");
  const [note, setNote] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (mode === "summarize" && !catalog) api.videoCatalog().then(setCatalog).catch((caught) => setError(caught instanceof Error ? caught.message : "Video catalog failed."));
  }, [mode, catalog]);

  async function search(event: FormEvent) {
    event.preventDefault(); if (!query.trim()) return;
    setLoading(true); setError(""); setSelected(null); setFollowUp(null); setMessage("");
    try { setResult(await api.videoMemory(query)); } catch (caught) { setError(caught instanceof Error ? caught.message : "Video search failed."); } finally { setLoading(false); }
  }

  async function summarize(event: FormEvent) {
    event.preventDefault(); if (!videoId) return;
    setLoading(true); setError(""); setSelected(null); setFollowUp(null); setMessage("");
    try { setSummary(await api.summarizeVideo(videoId)); } catch (caught) { setError(caught instanceof Error ? caught.message : "Video summary failed."); } finally { setLoading(false); }
  }

  function choose(item: { videoId: string; start: number; end: number; label: string; ids?: string[] }) {
    setSelected({ ...item, ids: item.ids ?? [] }); setFollowUp(null); setMessage("");
  }

  async function ask(event: FormEvent) {
    event.preventDefault(); if (!selected || !question.trim()) return;
    setLoading(true); setError("");
    try { setFollowUp(await api.videoFollowUp(selected.videoId, question, Math.max(0, selected.start - 2), selected.end + 2)); } catch (caught) { setError(caught instanceof Error ? caught.message : "Follow-up failed."); } finally { setLoading(false); }
  }

  async function save(event: FormEvent) {
    event.preventDefault(); if (!selected || !followUp) return;
    setLoading(true); setError("");
    try { await api.createVideoFinding({ video_id: selected.videoId, question: followUp.question, start_s: selected.start, end_s: selected.end, answer: followUp.answer, evidence_window_ids: followUp.evidence_window_ids, status, note, limitations: followUp.limitations, source: followUp.source }); setMessage("Finding saved to video history."); } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not save finding."); } finally { setLoading(false); }
  }

  return <>
    <section className={styles.hero}><div><p className="eyebrow">Video memory</p><h1>Find evidence in a recording.</h1><p>Find a specific moment or view a readable account of what happened. Every result stays tied to playable evidence.</p></div><aside className={styles.heroNote}><strong>Two simple ways to use it</strong><p>Find a moment when you know what to ask. Summarize a recording when you want an ordered account.</p></aside></section>
    <div className={`panel ${styles.searchPanel}`}><div className={styles.modeRow}><div className={styles.segments} role="tablist" aria-label="Video task"><button type="button" className={mode === "find" ? styles.active : ""} onClick={() => setMode("find")}>Find a moment</button><button type="button" className={mode === "summarize" ? styles.active : ""} onClick={() => setMode("summarize")}>Summarize a recording</button></div></div>{mode === "find" ? <form onSubmit={search}><div className={styles.question}><textarea aria-label="Video memory question" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="What would you like to find in a recording?" /><button className={styles.primary} disabled={loading || !query.trim()}>{loading ? "Searching…" : "Find moment"}</button></div><div className={styles.examples}>{["When did the person open the door?", "When did the person sit down?", "When was the book visible?"].map((example) => <button type="button" key={example} onClick={() => setQuery(example)}>{example}</button>)}</div></form> : <form onSubmit={summarize}><label className={styles.selectField}>Choose a prepared recording<select aria-label="Video to summarize" value={videoId} onChange={(event) => setVideoId(event.target.value)}><option value="">Choose a recording</option>{catalog?.videos.map((video) => <option key={video.video_id} value={video.video_id}>{video.video_id} · {video.duration_s.toFixed(1)} seconds</option>)}</select></label><button className={styles.primary} disabled={loading || !videoId}>{loading ? "Preparing…" : "Summarize recording"}</button></form>}{error && <p className="error" role="alert">{error}</p>}</div>
    {mode === "find" && result && <section className={styles.results} aria-live="polite"><div className={`panel ${styles.answerStrip}`}><div><span className="status strong">{result.retrieval_mode === "learned_temporal_clip" ? "learned video retrieval" : "annotation baseline"}</span><h2>{result.results.length} distinct candidate moments</h2><p>{result.catalog_window_count} prepared windows; {result.indexed_window_count} training windows indexed.</p></div><div className={styles.timeNotice}>Select a card to inspect its context and ask a follow-up.</div></div><div className={styles.evidenceGrid}>{result.results.map((item) => <article className={`panel ${styles.evidenceCard} ${selected?.videoId === item.video_id && selected.start === item.start_s ? styles.selectedEvidence : ""}`} key={item.window_id} onClick={() => choose({ videoId: item.video_id, start: item.start_s, end: item.end_s, label: item.actions.map((action) => action.name).join(" · ") || "Selected moment", ids: [item.window_id] })}>{item.video_url && <video controls preload="metadata" src={`${item.video_url}#t=${item.start_s},${item.end_s}`} onClick={(event) => event.stopPropagation()} />}<div className={styles.evidenceBody}><div className={styles.evidenceTop}><span className={styles.rank}>{item.video_id}</span><span className={styles.score}>{item.start_s.toFixed(1)}–{item.end_s.toFixed(1)} s</span></div><strong>{item.actions.map((action) => action.name).join(" · ") || "Relevant video window"}</strong><small>Select to inspect this moment and ask a follow-up.</small><small>Objects: {item.objects.join(" · ") || "none listed"}</small>{item.description && <p>{item.description}</p>}</div></article>)}</div>{result.results.length === 0 && <div className="panel"><p>No matching window was found. This is a safe no-result, not proof that the event never happened.</p></div>}{selected && <EvidenceReview selected={selected} question={question} setQuestion={setQuestion} followUp={followUp} loading={loading} error={error} message={message} status={status} setStatus={setStatus} note={note} setNote={setNote} ask={ask} save={save} />}</section>}
    {mode === "summarize" && summary && <section className={styles.results} aria-live="polite"><div className={`panel ${styles.answerStrip}`}><div><span className="status strong">recording overview</span><h2>{summary.video_id}</h2><p>{summary.overview || "The recording has official action annotations."}</p></div><div className={styles.timeNotice}>The overview is a broad dataset description. Timed actions below are the events with explicit time labels.</div></div><div className={`panel ${styles.timeline}`}><video controls preload="metadata" src={summary.video_url} /><h2>Timed actions</h2><p className={styles.timelineNote}>This list is not a complete description of every visible action. It contains only the actions the dataset marked with time intervals.</p>{summary.events.map((item) => <button type="button" className={styles.timelineEvent} key={`${item.label}-${item.start_s}`} onClick={() => choose({ videoId: summary.video_id, start: item.start_s, end: item.end_s, label: item.label, ids: item.source_events?.map((event) => event.evidence_window_id) })}><span>{item.start_s.toFixed(1)}–{item.end_s.toFixed(1)} s</span><strong>{item.label}</strong><small>Grouped from {item.source_events?.length ?? 1} source annotation(s).</small></button>)}<details><summary>Show source annotations</summary>{summary.raw_events.map((item) => <p key={`${item.label}-${item.start_s}`}>{item.start_s.toFixed(1)}–{item.end_s.toFixed(1)} s — {item.label}</p>)}</details></div>{selected && <EvidenceReview selected={selected} question={question} setQuestion={setQuestion} followUp={followUp} loading={loading} error={error} message={message} status={status} setStatus={setStatus} note={note} setNote={setNote} ask={ask} save={save} />}</section>}
  </>;
}

function EvidenceReview(props: { selected: { videoId: string; start: number; end: number; label: string; ids: string[] }; question: string; setQuestion: (value: string) => void; followUp: VideoFollowUp | null; loading: boolean; error: string; message: string; status: FindingStatus; setStatus: (value: FindingStatus) => void; note: string; setNote: (value: string) => void; ask: (event: FormEvent) => void; save: (event: FormEvent) => void }) {
  const { selected } = props;
  return <section className={`panel ${styles.followUp}`}><span className="eyebrow">Selected evidence</span><h2>{selected.label}</h2><video controls preload="metadata" src={`/api/video-memory/videos/${selected.videoId}#t=${Math.max(0, selected.start - 2)},${selected.end + 2}`} /><p>Reviewing {selected.start.toFixed(1)}–{selected.end.toFixed(1)} seconds with immediate context before and after.</p><form onSubmit={props.ask}><div className={styles.question}><textarea aria-label="Follow-up question" value={props.question} onChange={(event) => props.setQuestion(event.target.value)} placeholder="Ask about this moment" /><button className={styles.primary} disabled={props.loading || !props.question.trim()}>{props.loading ? "Checking…" : "Ask"}</button></div></form>{props.followUp && <form className={styles.analysis} onSubmit={props.save}><span className={`status ${props.followUp.supported ? "supported" : "uncertain"}`}>{props.followUp.supported ? "evidence-supported" : "unclear"}</span><p>{props.followUp.answer}</p><small>{props.followUp.limitations[0]}</small><label>Status<select value={props.status} onChange={(event) => props.setStatus(event.target.value as FindingStatus)}><option value="unclear">Unclear</option><option value="confirmed">Confirmed</option><option value="needs_manual_review">Needs manual review</option><option value="rejected">Rejected</option></select></label><textarea aria-label="Finding note" value={props.note} onChange={(event) => props.setNote(event.target.value)} placeholder="Optional note for the inspection record" /><button className={styles.secondary} disabled={props.loading}>Save finding</button>{props.message && <p className="success">{props.message}</p>}</form>}{props.error && <p className="error">{props.error}</p>}</section>;
}
