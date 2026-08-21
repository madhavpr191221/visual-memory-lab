import type {
  AnalysisResponse,
  Capabilities,
  QueryPage,
  TechnicianBenchmark,
  InspectionRecord,
  InspectionComparison,
  InspectionReport,
  VisualSummary,
  SearchResponse,
  GuidedDemo,
  Zone,
  Phase6b1Showcase,
  Phase612Showcase,
  Phase613Showcase,
  VideoMemoryResponse,
  VideoCatalogResponse,
  VideoSummary,
  VideoFollowUp,
  VideoGroundedAnswer,
  VideoObjectEvidence,
  VideoFinding,
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
      else if (body.detail) message = `Request validation failed: ${JSON.stringify(body.detail)}`;
    } catch {
      // Preserve the HTTP fallback message.
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export const api = {
  guidedDemo: () => request<GuidedDemo>("/api/guided-demo"),
  videoMemory: (query: string, videoId: string) => request<VideoMemoryResponse>(`/api/video-memory?q=${encodeURIComponent(query)}&video_id=${encodeURIComponent(videoId)}`),
  videoCatalog: () => request<VideoCatalogResponse>("/api/video-memory/catalog"),
  summarizeVideo: (videoId: string) => request<VideoSummary>("/api/video-memory/summarize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ video_id: videoId }) }),
  videoFollowUp: (videoId: string, question: string, startS: number, endS: number) => request<VideoFollowUp>("/api/video-memory/follow-up", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ video_id: videoId, question, start_s: startS, end_s: endS }) }),
  synthesizeVideo: (payload: { video_id: string; question: string; event_label: string; start_s: number; end_s: number; evidence_window_ids: string[]; mode: "preview" | "detailed" }) => request<VideoGroundedAnswer>("/api/video-memory/synthesize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  videoObjectEvidence: (payload: { video_id: string; query: string; event_label: string; start_s: number; end_s: number; object_prompts?: string[]; frame_timestamps_s?: number[] }) => request<VideoObjectEvidence>("/api/video-memory/object-evidence", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  videoFindings: () => request<VideoFinding[]>("/api/video-memory/findings"),
  createVideoFinding: (payload: Omit<VideoFinding, "id" | "created_at">) => request<VideoFinding>("/api/video-memory/findings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  videoFinding: (id: string) => request<VideoFinding>(`/api/video-memory/findings/${encodeURIComponent(id)}`),
  capabilities: () => request<Capabilities>("/api/capabilities"),
  searchText: (question: string, displayK: 3 | 5 | 10) =>
    request<SearchResponse>("/api/search/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, display_k: displayK }),
    }),
  searchImage: (image: File, displayK: 3 | 5 | 10) => {
    const body = new FormData();
    body.append("image", image);
    body.append("display_k", String(displayK));
    return request<SearchResponse>("/api/search/image", { method: "POST", body });
  },
  evaluation: () => request<Record<string, unknown>>("/api/memory/evaluation"),
  objects: () => request<Phase6b1Showcase>("/api/objects"),
  evidence: () => request<Phase612Showcase>("/api/evidence"),
  associations: () => request<Phase613Showcase>("/api/associations"),
  zones: () => request<Zone[]>("/api/zones"),
  zone: (slug: string) => request<Zone>(`/api/zones/${encodeURIComponent(slug)}`),
  queries: (offset: number, tag: string) => {
    const params = new URLSearchParams({ offset: String(offset), limit: "24" });
    if (tag) params.set("tag", tag);
    return request<QueryPage>(`/api/queries?${params}`);
  },
  query: (id: string) =>
    request<Record<string, unknown>>(`/api/queries/${encodeURIComponent(id)}`),
  technicianBenchmark: () => request<TechnicianBenchmark>("/api/technician-benchmark"),
  inspections: () => request<InspectionRecord[]>("/api/inspections"),
  inspection: (id: string) => request<InspectionRecord>(`/api/inspections/${encodeURIComponent(id)}`),
  createInspection: (title: string, question: string, evidenceIds: string[]) => request<InspectionRecord>("/api/inspections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title, question, evidence_ids: evidenceIds }) }),
  createInspectionWithImage: (title: string, question: string, evidenceIds: string[], image: File) => {
    const body = new FormData();
    body.append("title", title); body.append("question", question); body.append("evidence_ids", JSON.stringify(evidenceIds)); body.append("image", image);
    return request<InspectionRecord>("/api/inspections/with-image", { method: "POST", body });
  },
  compareInspection: (id: string, earlierObservationId: string) => request<InspectionComparison>(`/api/inspections/${encodeURIComponent(id)}/compare`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ earlier_observation_id: earlierObservationId }) }),
  summarizeInspectionImage: (image: File) => { const body = new FormData(); body.append("image", image); return request<VisualSummary>("/api/inspection-summary/image", { method: "POST", body }); },
  saveInspectionSummary: (id: string, summary: VisualSummary) => request<InspectionRecord>(`/api/inspections/${encodeURIComponent(id)}/summary`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(summary) }),
  inspectionReport: (id: string, question: string, earlierObservationId: string) => request<InspectionReport>(`/api/inspections/${encodeURIComponent(id)}/report`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, earlier_observation_id: earlierObservationId }) }),
  analyzeText: (question: string, evidenceIds: string[]) =>
    request<AnalysisResponse>("/api/analyze/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, evidence_ids: evidenceIds }),
    }),
  analyzeImage: (question: string, evidenceIds: string[], image: File) => {
    const body = new FormData();
    body.append("question", question);
    body.append("evidence_ids", JSON.stringify(evidenceIds));
    body.append("image", image);
    return request<AnalysisResponse>("/api/analyze/image", { method: "POST", body });
  },
};
