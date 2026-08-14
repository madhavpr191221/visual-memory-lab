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

export interface VideoMemoryWindow {
  window_id: string;
  video_id: string;
  video_path: string;
  video_url?: string;
  split: string;
  start_s: number;
  end_s: number;
  score?: number;
  retrieval_mode?: string;
  actions: { action_id: string; name: string; start_s: number; end_s: number }[];
  objects: string[];
  description: string;
}

export interface VideoMemoryResponse {
  dataset: string;
  window_count: number;
  query: string;
  retrieval_mode: string;
  results: VideoMemoryWindow[];
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
  result_kind?: string;
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
  retrieval_mode?: string;
  candidate_count?: number;
  diversity_note?: string;
}

export interface GuidedDemo {
  case_id: string;
  title: string;
  question: string;
  current: EvidenceItem;
  earlier: EvidenceItem;
  supporting_evidence: EvidenceItem[];
  outcome: string;
  explanation: string;
  manual_check: string;
  limitations: string[];
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

export interface TechnicianQuestion {
  question_id: string;
  question: string;
  category: string;
  dataset: string;
  answerability: "supported" | "supported_with_limits" | "requires_manual_review" | "unsupported";
  rationale: string;
  source_observation_id?: string;
  expected_zone?: string | null;
  expected_visit?: string | null;
  expected_object_class?: string | null;
  expected_artifact?: string | null;
}

export interface TechnicianBenchmark {
  question_count: number;
  questions: TechnicianQuestion[];
  summary: Record<string, unknown> | null;
}

export interface InspectionRecord {
  id: string;
  title: string;
  question: string;
  created_at: string;
  status: string;
  result_text: string;
  limitations: string[];
  selected_earlier_observation_id: string | null;
  evidence?: { observation_id: string; role: string; rank: number | null }[];
  summary_json?: VisualSummary | null;
  report_json?: InspectionReport | null;
}

export interface InspectionComparisonSide {
  image_url: string | null;
  observation_id: string | null;
  sequence_id: string | null;
  frame: number | null;
  zone: ZoneSummary | null;
  label: string;
}

export interface InspectionComparison {
  inspection_id: string;
  current: InspectionComparisonSide;
  earlier: InspectionComparisonSide;
  status: string;
  explanation: string;
  limitations: string[];
  updated_inspection: InspectionRecord;
}

export interface VisualSummary {
  summary: string;
  visible_objects: string[];
  visible_conditions: string[];
  limitations: string[];
  model: string;
  cached: boolean;
}

export interface InspectionReport {
  status: "observed" | "possible_difference" | "insufficient_evidence" | "manual_review_required";
  summary: string;
  visible_objects: string[];
  visible_conditions: string[];
  comparison_observations: string[];
  supporting_evidence: { observation_id: string; claim: string }[];
  limitations: string[];
  recommended_manual_check: string;
  model: string;
  cached: boolean;
  inspection: InspectionRecord;
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

export interface ChangeFocusBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ChangeCase {
  pair_id: string;
  earlier_observation: number;
  current_observation: number;
  earlier_image_url: string;
  current_image_url: string;
  earlier_frame: number;
  current_frame: number;
  earlier_box: ChangeFocusBox;
  current_box: ChangeFocusBox;
  outcome: "object_added" | "object_removed" | "object_moved" | "uncertain" | "reconstruction_artifact";
  outcome_label: string;
  headline: string;
  confidence: "low" | "medium" | "high";
  explanation: string;
  limitation: string;
  geometry_url: string;
  geometry_note: string;
}

export interface Phase6aShowcase {
  dataset: string;
  logical_order_note: string;
  claim_boundary: string;
  vlm_audit?: { verdict: "same" | "different" | "uncertain"; explanation: string };
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
  cases: ChangeCase[];
}

export type ObjectClass = "chair" | "waste_bin" | "box";
export type ObjectAuditStatus = "supported" | "uncertain" | "unsupported" | "unreviewed";

export interface ObjectDetectionAudit {
  detection_id: string;
  verdict: Exclude<ObjectAuditStatus, "unreviewed">;
  category_correct: "yes" | "no" | "uncertain";
  mask_quality: "good" | "partial" | "excessive" | "uncertain";
  explanation: string;
}

export interface ObjectDetection {
  detection_id: string;
  frame_id: string;
  canonical_class: ObjectClass;
  phrase: string;
  score: number;
  box_xyxy: [number, number, number, number];
  box_normalized: [number, number, number, number];
  mask_url: string;
  mask_score: number;
  mask_area_fraction: number;
  warnings: string[];
  audit: ObjectDetectionAudit | null;
  audit_status: ObjectAuditStatus;
}

export interface ObjectFrame {
  frame_id: string;
  observation: number;
  message_index: number;
  timestamp_ns: number;
  width: number;
  height: number;
  pose: {
    frame: "T_G_C";
    translation_m: [number, number, number];
    quaternion_xyzw: [number, number, number, number];
  };
  image_url: string;
  overlay_url: string;
  detections: ObjectDetection[];
  audit_status: "reviewed" | "unreviewed";
  missed_visible_classes: ObjectClass[];
  audit_limitations: string[];
}

export interface ObjectAuditSummary {
  frame_count: number;
  reviewed_detection_count: number;
  verdict_counts: Record<string, number>;
  mask_quality_counts: Record<string, number>;
  missed_visible_class_counts: Record<string, number>;
  high_confidence_pseudo_support_rate: number | null;
  claim_boundary: string;
  model_requested: string;
  response_models: string[];
}

export interface Phase6b1Showcase {
  dataset: string;
  claim_boundary: string;
  metrics: {
    frame_count: number;
    detection_count: number;
    frames_with_detections: number;
    empty_frame_count: number;
    class_counts: Record<ObjectClass, number>;
    frames_per_observation: Record<string, number>;
  };
  method: {
    prompt: string;
    box_threshold: number;
    text_threshold: number;
    nms_iou: number;
    detector: Record<string, unknown>;
    segmenter: Record<string, unknown>;
  };
  audit: ObjectAuditSummary | null;
  frames: ObjectFrame[];
}

export interface RgbdEvidence {
  detection_id: string;
  frame_id: string;
  observation: number;
  message_index: number;
  canonical_class: ObjectClass;
  point_count: number;
  centroid_world_m: [number | null, number | null, number | null];
  extent_world_m: { minimum: [number | null, number | null, number | null]; maximum: [number | null, number | null, number | null] };
  evidence_method: string;
  claim_boundary: string;
  score: number;
  mask_score: number;
  mask_area_fraction: number;
  image_url: string;
  mask_url: string;
  width: number;
  height: number;
}

export interface RgbdComparison {
  id: string;
  earlier_observation: number;
  later_observation: number;
  object_class: ObjectClass;
  earlier: RgbdEvidence;
  later: RgbdEvidence;
  interpretation: string;
}

export interface Phase612Showcase {
  phase: string;
  dataset: string;
  claim_boundary: string;
  metrics: { frame_count: number; detection_count: number; evidence_count: number; nonempty_evidence_count: number };
  classes: ObjectClass[];
  comparisons: RgbdComparison[];
}

export interface AssociationSide extends ObjectDetection {
  observation: number;
  message_index: number;
  image_url: string;
  mask_url: string;
}

export interface AssociationPair {
  pair_id: string;
  earlier_observation: number;
  later_observation: number;
  object_class: ObjectClass;
  earlier_detection_id: string;
  later_detection_id: string;
  appearance_similarity: number;
  shape_score: number;
  evidence_score: number;
  position_score: number;
  association_score: number;
  centroid_distance_m: number | null;
  association_status: "likely_same" | "possible_match" | "uncertain";
  movement_status: "possible_movement" | "not_established";
  claim_boundary: string;
  vlm_audit?: { verdict: "same" | "different" | "uncertain"; explanation: string };
  earlier: AssociationSide;
  later: AssociationSide;
}

export interface Phase613Showcase {
  phase: string;
  claim_boundary: string;
  metrics: { pair_count: number; detection_count: number; classes: ObjectClass[] };
  classes: ObjectClass[];
  pairs: AssociationPair[];
}
