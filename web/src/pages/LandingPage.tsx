import { Link } from "react-router";
import styles from "./pages.module.css";

export function LandingPage() {
  return (
    <>
      <header className="page-heading">
        <span className="eyebrow">Office visual memory</span>
        <h1>Choose how you want to explore the evidence.</h1>
        <p>
          Use the office memory assistant to find earlier views, or inspect the
          measurements behind the system.
        </p>
      </header>
      <section className={styles.modeChoices} aria-label="Choose a workspace">
        <Link className={`panel ${styles.modeChoice}`} to="/app">
          <span className="eyebrow">For technicians and users</span>
          <h2>Use Visual Memory</h2>
          <p>Ask where something was seen and inspect the images that support the answer.</p>
          <span className={styles.modeAction}>Open the assistant →</span>
        </Link>
        <Link className={`panel ${styles.modeChoice} ${styles.researchChoice}`} to="/research">
          <span className="eyebrow">For reviewers and engineers</span>
          <h2>System Insights</h2>
          <p>Review retrieval quality, failures, zones, object predictions, and 3D evidence.</p>
          <span className={styles.modeAction}>Inspect the system →</span>
        </Link>
      </section>
    </>
  );
}
