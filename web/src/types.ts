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

export interface ChangeCandidateReview {
  candidate_id: string;
  verdict: "supported" | "unsupported" | "uncertain";
  interpretation: "current_only" | "earlier_only" | "possible_move" | "unknown";
  description: string;
  confidence: "low" | "medium" | "high";
  evidence_ids: string[];
  limitations: string[];
  related_candidate_id: string | null;
}

export interface ChangeObservation {
  observation_id: string;
  logical_order: number;
  frame_count: number;
  frames: { message_index: number; timestamp_ns: number; image_url: string }[];
  contact_sheet_url: string;
  vlm_contact_sheet_url: string;
}

export interface ChangePair {
  pair_id: string;
  earlier_observation: number;
  current_observation: number;
  consecutive: boolean;
  current_only_candidate_count: number;
  earlier_only_candidate_count: number;
  current_only_projection_url: string;
  earlier_only_projection_url: string;
  changed_fraction: Record<string, { current_only: number; earlier_only: number }>;
  point_to_point: Record<string, unknown>;
  reviewed_candidates: ChangeCandidateReview[];
  review_limitations: string[];
}

export interface Phase6aShowcase {
  dataset: string;
  logical_order_note: string;
  claim_boundary: string;
  metrics: {
    observation_count: number;
    rgb_sample_count: number;
    pair_count: number;
    geometric_candidate_count: number;
    reviewed_candidate_count: number;
    accepted_pseudo_reference_count: number;
    verdict_counts: { supported: number; uncertain: number; unsupported: number };
  };
  method: {
    voxel_size_m: number;
    primary_threshold_m: number;
    distance_thresholds_m: number[];
    min_cluster_voxels: number;
  };
  observations: ChangeObservation[];
  pairs: ChangePair[];
}
