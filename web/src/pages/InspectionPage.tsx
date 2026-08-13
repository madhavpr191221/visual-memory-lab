import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../api";
import type { InspectionComparison, InspectionRecord, InspectionReport, SearchResponse, VisualSummary } from "../types";
import styles from "./pages.module.css";

export function InspectionPage() {
  const [question, setQuestion] = useState("Where was this office area seen before?");
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"text" | "image">("text");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [earlier, setEarlier] = useState<string | null>(null);
  const [saved, setSaved] = useState<InspectionRecord | null>(null);
  const [comparison, setComparison] = useState<InspectionComparison | null>(null);
  const [summary, setSummary] = useState<VisualSummary | null>(null);
  const [summaryError, setSummaryError] = useState("");
  const [report, setReport] = useState<InspectionReport | null>(null);
  const [reporting, setReporting] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  async function search(event?: FormEvent) {
    event?.preventDefault(); setLoading(true); setError(""); setSummaryError(""); setComparison(null); setReport(null); setSummary(null); setSaved(null);
    try {
      const response = mode === "image" && file ? await api.searchImage(file, 5) : await api.searchText(question, 5);
      setResult(response); setSelected(response.evidence.slice(0, 5).map((item) => item.observation_id)); setEarlier(null);
      if (mode === "image" && file) {
        try { setSummary(await api.summarizeInspectionImage(file)); }
        catch (reason) { setSummaryError(reason instanceof Error ? reason.message : "Visual summary unavailable."); }
      }
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Search failed."); }
    finally { setLoading(false); }
  }
  async function save() {
    if (!result) return;
    try {
      let record = mode === "image" && file ? await api.createInspectionWithImage("Office inspection", question, selected, file) : await api.createInspection("Office inspection", question, selected);
      if (summary) record = await api.saveInspectionSummary(record.id, summary);
      setSaved(record);
      if (earlier) setComparison(await api.compareInspection(record.id, earlier));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not save inspection."); }
  }
  async function compare() {
    if (!saved || !earlier) return;
    try { setComparison(await api.compareInspection(saved.id, earlier)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not compare views."); }
  }
  async function generateReport() {
    if (!saved || !earlier) return;
    setReporting(true); setError("");
    try { setReport(await api.inspectionReport(saved.id, question, earlier)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not generate inspection report."); }
    finally { setReporting(false); }
  }
  useEffect(() => { void search(); }, []);
  return <>
    <header className="page-heading"><span className="eyebrow">Office inspection</span><h1>Find and explain office evidence.</h1><p>Ask a question, choose an earlier view, and compare it with the current office view.</p></header>
    <form className={`panel ${styles.searchPanel}`} onSubmit={search}><div className={styles.segments}><button type="button" className={mode === "text" ? styles.active : ""} onClick={() => setMode("text")}>Ask with text</button><button type="button" className={mode === "image" ? styles.active : ""} onClick={() => setMode("image")}>Use current image</button></div>{mode === "image" && <label className={styles.upload}>Choose a PNG or JPEG under 10 MB<input type="file" accept="image/png,image/jpeg" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></label>}<textarea aria-label="Inspection question" value={question} onChange={(event) => setQuestion(event.target.value)} /><div className={styles.confirmActions}><button className={styles.primary} type="submit" disabled={loading || (mode === "image" && !file)}>{loading ? "Finding evidence…" : "Find evidence"}</button><button className={styles.secondary} type="button" onClick={save} disabled={!result}>{earlier ? "Save and compare" : "Save inspection"}</button></div>{error && <p className="error">{error}</p>}</form>
    {result && <section className={styles.results}><div className={styles.resultHeader}><h2>Possible earlier views</h2><span>{earlier ? "1 earlier view chosen" : "Choose one to compare"}</span></div><div className={styles.evidenceGrid}>{result.evidence.slice(0, 5).map((item) => <article className={`panel ${styles.evidenceCard} ${earlier === item.observation_id ? styles.selectedEvidence : ""}`} key={item.observation_id}><img src={item.image_url} alt={item.zone?.name ?? "Office evidence"} /><div className={styles.evidenceBody}><strong>{item.zone?.name ?? "Unassigned area"}</strong><small>{item.sequence_id} · frame {item.frame} · CLIP {item.score.toFixed(3)}</small><button type="button" onClick={() => setEarlier(item.observation_id)}>{earlier === item.observation_id ? "Earlier view chosen" : "Use as earlier view"}</button></div></article>)}</div></section>}
    {summary && <section className={`panel ${styles.inspectionSummary}`}><span className="eyebrow">What I can see</span><h2>Current photo summary</h2><p>{summary.summary}</p><div className={styles.summaryColumns}><div><strong>Visible objects</strong><p>{summary.visible_objects.join(" · ") || "None confidently identified"}</p></div><div><strong>Visible conditions</strong><p>{summary.visible_conditions.join(" · ") || "No specific condition noted"}</p></div></div>{summary.limitations.length > 0 && <small>Limits: {summary.limitations.join(" ")}</small>}</section>}
    {summaryError && <section className={`panel ${styles.inspectionSummary}`}><strong>Visual summary unavailable</strong><p>{summaryError} Retrieval can still continue.</p></section>}
    {saved && <section className={`panel ${styles.analysis}`}><span className="status moderate">saved locally</span><h2>Inspection ready</h2><p>{saved.result_text}</p>{earlier && !comparison && <button className={styles.primary} type="button" onClick={compare}>Compare with earlier view</button>}<Link to="/app/inspections">Open inspection history →</Link></section>}
    {comparison && <section className={`panel ${styles.inspectionComparison}`}><div className={styles.resultHeader}><div><span className="eyebrow">Side-by-side review</span><h2>Current view and earlier view</h2></div><span className="status moderate">{comparison.status.replaceAll("_", " ")}</span></div><div className={styles.inspectionSides}>{[comparison.current, comparison.earlier].map((side) => <article className={styles.inspectionSide} key={side.label}><h3>{side.label}</h3>{side.image_url ? <img src={side.image_url} alt={side.label} /> : <div className={styles.missingImage}>No image available</div>}<p>{side.zone?.name ?? "Zone unavailable"} · {side.sequence_id ?? "uploaded photo"}{side.frame !== null ? ` · frame ${side.frame}` : ""}</p></article>)}</div><div className={styles.inspectionConclusion}><strong>What we can safely say</strong><p>{comparison.explanation}</p><small>{comparison.limitations.join(" ")}</small></div>{saved && <button className={styles.primary} type="button" onClick={generateReport} disabled={reporting}>{reporting ? "Preparing inspection report…" : "Generate inspection report"}</button>}</section>}
    {report && <section className={`panel ${styles.inspectionReport}`}><div className={styles.resultHeader}><div><span className="eyebrow">Technician report</span><h2>{report.status.replaceAll("_", " ")}</h2></div><span className="status moderate">Evidence review</span></div><p>{report.summary}</p><div className={styles.summaryColumns}><div><strong>Visible objects</strong><p>{report.visible_objects.join(" · ") || "None confidently identified"}</p></div><div><strong>Visible conditions</strong><p>{report.visible_conditions.join(" · ") || "No specific condition noted"}</p></div></div><h3>What differs or needs checking</h3><ul>{report.comparison_observations.map((item) => <li key={item}>{item}</li>)}</ul><div className={styles.inspectionConclusion}><strong>Recommended manual check</strong><p>{report.recommended_manual_check}</p></div>{report.limitations.length > 0 && <small>Limits: {report.limitations.join(" ")}</small>}</section>}
  </>;
}

export function InspectionsPage() {
  const [items, setItems] = useState<InspectionRecord[]>([]);
  useEffect(() => { api.inspections().then(setItems).catch(() => setItems([])); }, []);
  return <><header className="page-heading"><span className="eyebrow">Inspection history</span><h1>Saved office inspections</h1><p>Reopen a previous question and its selected evidence.</p></header><section className={styles.queryGrid}>{items.map((item) => <Link className={`panel ${styles.queryCard}`} to={`/app/inspections/${item.id}`} key={item.id}><div className={styles.queryBody}><strong>{item.title}</strong><p>{item.question}</p><small>{new Date(item.created_at).toLocaleString()} · {item.status}</small></div></Link>)}</section></>;
}

export function InspectionDetailPage({ id }: { id: string }) {
  const [item, setItem] = useState<InspectionRecord | null>(null);
  const [comparison, setComparison] = useState<InspectionComparison | null>(null);
  useEffect(() => { api.inspection(id).then(setItem).catch(() => setItem(null)); }, [id]);
  useEffect(() => { if (item?.selected_earlier_observation_id) void api.compareInspection(id, item.selected_earlier_observation_id).then(setComparison).catch(() => setComparison(null)); }, [id, item?.selected_earlier_observation_id]);
  if (!item) return <p>Loading inspection…</p>;
  return <><header className="page-heading"><span className="eyebrow">Saved inspection</span><h1>{item.title}</h1><p>{item.question}</p></header>{item.summary_json && <section className={`panel ${styles.inspectionSummary}`}><span className="eyebrow">What I can see</span><h2>Saved current-photo summary</h2><p>{item.summary_json.summary}</p></section>}<section className={`panel ${styles.analysis}`}><span className="status moderate">{item.status}</span><p>{item.result_text}</p><p><strong>Limits:</strong> {item.limitations.join(" ")}</p></section>{comparison && <section className={`panel ${styles.inspectionComparison}`}><div className={styles.resultHeader}><h2>Saved side-by-side review</h2><span className="status moderate">{comparison.status.replaceAll("_", " ")}</span></div><div className={styles.inspectionSides}>{[comparison.current, comparison.earlier].map((side) => <article className={styles.inspectionSide} key={side.label}><h3>{side.label}</h3>{side.image_url ? <img src={side.image_url} alt={side.label} /> : <div className={styles.missingImage}>No image available</div>}<p>{side.zone?.name ?? "Zone unavailable"} · {side.sequence_id ?? "uploaded photo"}{side.frame !== null ? ` · frame ${side.frame}` : ""}</p></article>)}</div><div className={styles.inspectionConclusion}><strong>What we can safely say</strong><p>{comparison.explanation}</p><small>{comparison.limitations.join(" ")}</small></div></section>}{item.report_json && <section className={`panel ${styles.inspectionReport}`}><span className="eyebrow">Technician report</span><h2>{item.report_json.status.replaceAll("_", " ")}</h2><p>{item.report_json.summary}</p><div className={styles.inspectionConclusion}><strong>Recommended manual check</strong><p>{item.report_json.recommended_manual_check}</p></div></section>}<section className="panel"><h2>Selected evidence</h2>{item.evidence?.map((e) => <p key={e.observation_id}>{e.observation_id}</p>)}</section></>;
}
