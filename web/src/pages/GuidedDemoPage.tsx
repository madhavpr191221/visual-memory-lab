import { useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../api";
import type { GuidedDemo } from "../types";
import styles from "./pages.module.css";

export function GuidedDemoPage() {
  const [demo, setDemo] = useState<GuidedDemo | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { api.guidedDemo().then(setDemo).catch((reason: Error) => setError(reason.message)); }, []);
  if (error) return <><header className="page-heading"><span className="eyebrow">Guided showcase</span><h1>The demo case is unavailable.</h1><p>{error}</p></header><Link className={styles.secondary} to="/app">Open the Office assistant →</Link></>;
  if (!demo) return <p>Loading guided case…</p>;
  return <>
    <header className="page-heading"><span className="eyebrow">90-second guided case</span><h1>{demo.title}</h1><p>A facilities technician wants a quick, evidence-backed answer before starting a manual workstation check.</p></header>
    <aside className={styles.claimBoundary}><strong>Demo boundary</strong><span>This is a guided presentation of real office evidence. It does not claim calendar time or persistent object identity.</span></aside>
    <section className={styles.demoFlow} aria-label="Guided case steps">{["Retrieve earlier evidence", "Compare the views", "Explain the safe conclusion", "Recommend the manual check"].map((step, index) => <div className="panel" key={step}><span>{index + 1}</span><strong>{step}</strong></div>)}</section>
    <section className={`panel ${styles.demoOutcome}`}><span className="status strong">Evidence-backed outcome</span><h2>{demo.outcome}</h2><p>{demo.explanation}</p></section>
    <section><div className={styles.resultHeader}><h2>Earlier and current evidence</h2><span>Selected from the real office memory index</span></div><div className={styles.compareGrid}>
      {[{ title: "Current reference view", item: demo.current }, { title: "Earlier retrieved view", item: demo.earlier }].map(({ title, item }) => <article className={`panel ${styles.changeFigure}`} key={title}><img src={item.image_url} alt={title} /><div className={styles.evidenceBody}><strong>{title}</strong><small>{item.zone?.name ?? "Office view"} · {item.sequence_id} · frame {item.frame} · CLIP {item.score.toFixed(3)}</small></div></article>)}
    </div></section>
    <section className={`panel ${styles.demoCheck}`}><h2>What should the technician check?</h2><p>{demo.manual_check}</p><h3>Why the system stays cautious</h3><ul>{demo.limitations.map((item) => <li key={item}>{item}</li>)}</ul></section>
    <nav className={styles.demoLinks} aria-label="Continue exploring"><Link className={styles.primaryLink} to="/app/inspect">Try your own inspection →</Link><Link className={styles.secondaryLink} to="/research">Review the measurements →</Link></nav>
  </>;
}
