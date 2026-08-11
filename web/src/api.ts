import type {
  AnalysisResponse,
  Capabilities,
  QueryPage,
  SearchResponse,
  Zone,
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
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Preserve the HTTP fallback message.
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export const api = {
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
  evaluation: () => request<Record<string, unknown>>("/api/evaluation"),
  zones: () => request<Zone[]>("/api/zones"),
  zone: (slug: string) => request<Zone>(`/api/zones/${encodeURIComponent(slug)}`),
  queries: (offset: number, tag: string) => {
    const params = new URLSearchParams({ offset: String(offset), limit: "24" });
    if (tag) params.set("tag", tag);
    return request<QueryPage>(`/api/queries?${params}`);
  },
  query: (id: string) =>
    request<Record<string, unknown>>(`/api/queries/${encodeURIComponent(id)}`),
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
