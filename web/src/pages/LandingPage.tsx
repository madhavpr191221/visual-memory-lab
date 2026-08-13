import { Link } from "react-router";
import styles from "./pages.module.css";

export function LandingPage() {
  return <>
    <header className="page-heading">
      <span className="eyebrow">Office visual memory</span>
      <h1>What do you want to do?</h1>
      <p>Use the office assistant to answer a practical question, watch a guided case, or open the research workspace to inspect how the system works.</p>
    </header>
    <section className={styles.modeChoices} aria-label="Choose a workspace">
      <article className={`panel ${styles.modeChoice}`}>
        <span className="eyebrow">For technicians and users</span>
        <h2>Office assistant</h2>
        <p>Ask where something was seen and inspect the images that support the answer.</p>
        <Link className={styles.modeAction} to="/app">Open the assistant →</Link>
        <Link className={styles.modeAction} to="/app/demo">Watch the guided case →</Link>
      </article>
      <Link className={`panel ${styles.modeChoice} ${styles.researchChoice}`} to="/research">
        <span className="eyebrow">For reviewers and engineers</span>
        <h2>Research workspace</h2>
        <p>Review retrieval quality, failures, zones, object predictions, and 3D evidence.</p>
        <span className={styles.modeAction}>Inspect the system →</span>
      </Link>
    </section>
  </>;
}
