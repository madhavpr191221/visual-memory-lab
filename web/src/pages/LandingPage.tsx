import { Link } from "react-router";
import styles from "./pages.module.css";

export function LandingPage() {
  return <>
    <header className="page-heading">
      <span className="eyebrow">Video memory</span>
      <h1>Find the moment you need.</h1>
      <p>Choose a recording, ask a plain-language question, and review the playable evidence before saving a finding.</p>
    </header>
    <section className={styles.modeChoices} aria-label="Choose a workspace">
      <Link className={`panel ${styles.modeChoice}`} to="/app">
        <span className="eyebrow">For technicians and reviewers</span>
        <h2>Open video memory</h2>
        <p>Search one recording, inspect a moment, ask a follow-up, and save what the evidence supports.</p>
        <span className={styles.modeAction}>Start with a recording →</span>
      </Link>
      <Link className={`panel ${styles.modeChoice} ${styles.researchChoice}`} to="/research">
        <span className="eyebrow">For engineers and evaluators</span>
        <h2>Open Research</h2>
        <p>Inspect retrieval quality, event boundaries, failures, and the older office/image experiments.</p>
        <span className={styles.modeAction}>Inspect the system →</span>
      </Link>
    </section>
  </>;
}
