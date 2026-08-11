export type EvidenceStrength = "strong" | "moderate" | "mixed";

export interface Capabilities {
  dataset: string;
  memory_count: number;
  query_count: number;
  captured_at_available: boolean;
  analysis_available: boolean;
  analysis_requires_confirmation: boolean;
  search_modes: string[];
  supported_question_families: string[];
  unsupported_claims: string[];
}

export interface ZoneSummary {
  slug: string;
  name: string;
}

export interface EvidenceItem {
  rank: number;
  score: number;
  observation_id: string;
  collection: "memory" | "query";
  sequence_id: string | null;
  frame: number | null;
  captured_at: null;
  zone: ZoneSummary | null;
  image_url: string;
}

export interface SearchResponse {
  query: { kind: "text" | "image"; question: string | null };
  display_k: 3 | 5 | 10;
  temporal: { captured_at: null; message: string };
  likely_area: {
    slug: string;
    name: string;
    support_count: number;
    evidence_count: number;
    strength: EvidenceStrength;
  } | null;
  evidence: EvidenceItem[];
}

export interface AnalysisResponse {
  question_type: string;
  supported: boolean;
  answer: string;
  evidence_citations: { observation_id: string; claim: string }[];
  evidence_strength: "low" | "medium" | "high";
  limitations: string[];
  model: string;
  cached: boolean;
}

export interface Zone {
  slug: string;
  name: string;
  description: string;
  stable_landmarks: string[];
  prompts: Record<string, string>;
  assigned_frame_count: number;
  memories?: { observation_id: string; image_url: string }[];
}

export interface QueryListItem {
  query_id: string;
  sequence_id: string;
  frame: number;
  image_url: string;
  top1_translation_error_m: number;
  top1_rotation_error_deg: number;
  tags: string[];
}

export interface QueryPage {
  offset: number;
  limit: number;
  total: number;
  items: QueryListItem[];
}
