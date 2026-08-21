import { Navigate, NavLink, Outlet, Route, Routes, useLocation } from "react-router";
import { AssociationPage } from "./pages/AssociationPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { FailuresPage } from "./pages/FailuresPage";
import { HomePage } from "./pages/HomePage";
import { LandingPage } from "./pages/LandingPage";
import { GuidedDemoPage } from "./pages/GuidedDemoPage";
import { ObjectsPage } from "./pages/ObjectsPage";
import { QueryDetailPage } from "./pages/QueryDetailPage";
import { TechnicianTasksPage } from "./pages/TechnicianTasksPage";
import { InspectionDetailPage, InspectionPage, InspectionsPage } from "./pages/InspectionPage";
import { RgbdEvidencePage } from "./pages/RgbdEvidencePage";
import { ZoneDetailPage } from "./pages/ZoneDetailPage";
import { ZonesPage } from "./pages/ZonesPage";
import { VideoMemoryPage } from "./pages/VideoMemoryPage";
import { VideoHistoryPage } from "./pages/VideoHistoryPage";
import styles from "./App.module.css";

function Header() {
  const location = useLocation();
  const research = location.pathname.startsWith("/research");
  return (
    <header className={styles.header}>
      <NavLink className={styles.brand} to="/">
        <span className={styles.mark} aria-hidden="true">VM</span>
        <span><strong>Visual Memory Lab</strong><small>Evidence before answers</small></span>
      </NavLink>
      <nav aria-label={research ? "System insights navigation" : "Application navigation"}>
        {research ? <>
          <NavLink to="/app" className={styles.workspaceLink}>Application</NavLink>
          <NavLink to="/research" end>Overview</NavLink>
          <NavLink to="/research/evaluation">Evaluation</NavLink>
          <NavLink to="/research/failures">Failures</NavLink>
          <NavLink to="/research/zones">Zones</NavLink>
          <NavLink to="/research/objects">Objects</NavLink>
          <NavLink to="/research/evidence">3D evidence</NavLink>
          <NavLink to="/research/associations">Associations</NavLink>
        </> : <>
          <NavLink to="/app" end>Video memory</NavLink>
          <NavLink to="/app/video-history">Saved findings</NavLink>
          <NavLink to="/archive/office">Office archive</NavLink>
          <NavLink to="/research" className={styles.workspaceLink}>Research</NavLink>
        </>}
      </nav>
    </header>
  );
}

function Shell() {
  return <div className={styles.shell}><Header /><main className={styles.main}><Outlet /></main><footer className={styles.footer}>Public office data | Visual memory | Evidence before answers</footer></div>;
}

export default function App() {
  return <Routes>
    <Route element={<Shell />}>
      <Route index element={<LandingPage />} />
      <Route path="app" element={<VideoMemoryPage />} />
      <Route path="app/demo" element={<GuidedDemoPage />} />
      <Route path="app/objects" element={<ObjectsPage />} />
      <Route path="app/evidence" element={<RgbdEvidencePage />} />
      <Route path="app/compare" element={<AssociationPage />} />
      <Route path="app/tasks" element={<TechnicianTasksPage />} />
      <Route path="app/inspect" element={<InspectionPage />} />
      <Route path="app/video" element={<VideoMemoryPage />} />
      <Route path="app/video-history" element={<VideoHistoryPage />} />
      <Route path="archive/office" element={<HomePage />} />
      <Route path="archive/office/inspect" element={<InspectionPage />} />
      <Route path="archive/office/history" element={<InspectionsPage />} />
      <Route path="app/inspections" element={<InspectionsPage />} />
      <Route path="app/inspections/:id" element={<InspectionDetailPage id={window.location.pathname.split("/").pop() ?? ""} />} />
      <Route path="research" element={<EvaluationPage />} />
      <Route path="research/evaluation" element={<EvaluationPage />} />
      <Route path="research/failures" element={<FailuresPage />} />
      <Route path="research/queries/:queryId" element={<QueryDetailPage />} />
      <Route path="research/zones" element={<ZonesPage />} />
      <Route path="research/zones/:zoneSlug" element={<ZoneDetailPage />} />
      <Route path="research/objects" element={<ObjectsPage />} />
      <Route path="research/evidence" element={<RgbdEvidencePage />} />
      <Route path="research/associations" element={<AssociationPage />} />
      {/* Historical links from evidence cards redirect into the research workspace. */}
      <Route path="lab/evaluation" element={<Navigate to="/research/evaluation" replace />} />
      <Route path="lab/failures" element={<Navigate to="/research/failures" replace />} />
      <Route path="lab/queries/:queryId" element={<QueryDetailPage />} />
      <Route path="lab/zones" element={<Navigate to="/research/zones" replace />} />
      <Route path="lab/zones/:zoneSlug" element={<ZoneDetailPage />} />
    </Route>
  </Routes>;
}
