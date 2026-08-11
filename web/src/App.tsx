import { NavLink, Outlet, Route, Routes } from "react-router";
import { EvaluationPage } from "./pages/EvaluationPage";
import { FailuresPage } from "./pages/FailuresPage";
import { HomePage } from "./pages/HomePage";
import { QueryDetailPage } from "./pages/QueryDetailPage";
import { ZoneDetailPage } from "./pages/ZoneDetailPage";
import { ZonesPage } from "./pages/ZonesPage";
import styles from "./App.module.css";

function Layout() {
  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <NavLink className={styles.brand} to="/">
          <span className={styles.mark} aria-hidden="true">VM</span>
          <span>
            <strong>Office Visual Memory</strong>
            <small>Evidence before answers</small>
          </span>
        </NavLink>
        <nav aria-label="Main navigation">
          <NavLink to="/" end>Ask memory</NavLink>
          <NavLink to="/lab/evaluation">Evidence Lab</NavLink>
          <NavLink to="/lab/failures">Failures</NavLink>
          <NavLink to="/lab/zones">Zones</NavLink>
        </nav>
      </header>
      <main className={styles.main}><Outlet /></main>
      <footer className={styles.footer}>
        Public 7-Scenes Office data | Exact CLIP memory | Local retrieval
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="lab/evaluation" element={<EvaluationPage />} />
        <Route path="lab/failures" element={<FailuresPage />} />
        <Route path="lab/queries/:queryId" element={<QueryDetailPage />} />
        <Route path="lab/zones" element={<ZonesPage />} />
        <Route path="lab/zones/:zoneSlug" element={<ZoneDetailPage />} />
      </Route>
    </Routes>
  );
}
