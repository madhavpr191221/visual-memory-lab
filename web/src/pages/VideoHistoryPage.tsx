import { useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../api";
import type { VideoFinding } from "../types";
import styles from "./pages.module.css";

export function VideoHistoryPage() {
  const [items, setItems] = useState<VideoFinding[]>([]);
  useEffect(() => { api.videoFindings().then(setItems).catch(() => setItems([])); }, []);
  return <><header className="page-heading"><span className="eyebrow">Video history</span><h1>Saved video findings</h1><p>Reopen questions and evidence that you marked for later review.</p></header><section className={styles.queryGrid}>{items.map((item) => <article className={`panel ${styles.queryCard}`} key={item.id}><div className={styles.queryBody}><strong>{item.video_id} · {item.start_s.toFixed(1)}–{item.end_s.toFixed(1)} s</strong><p>{item.question}</p><p>{item.answer}</p><small>{new Date(item.created_at).toLocaleString()} · {item.status.replaceAll("_", " ")}</small><Link to={`/app/video?finding=${encodeURIComponent(item.id)}`}>Open video memory →</Link></div></article>)}{items.length === 0 && <div className="panel"><p>No saved video findings yet.</p></div>}</section></>;
}
