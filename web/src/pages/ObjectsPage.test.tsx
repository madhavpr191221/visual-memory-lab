import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ObjectsPage } from "./ObjectsPage";

const payload = {
  dataset: "ETH ASL Change Detection: Office",
  claim_boundary: "Detections and masks are predictions, not ground truth.",
  metrics: { frame_count: 384, detection_count: 1, frames_with_detections: 1, empty_frame_count: 383, class_counts: { chair: 1, waste_bin: 0, box: 0 }, frames_per_observation: { "0": 96, "1": 96, "2": 96, "3": 96 } },
  method: { prompt: "office chair.", box_threshold: .25, text_threshold: .2, nms_iou: .5, detector: { model_id: "grounding-dino-tiny" }, segmenter: { model_id: "sam2.1-hiera-small" } },
  audit: { frame_count: 1, reviewed_detection_count: 1, verdict_counts: { supported: 1 }, mask_quality_counts: { good: 1 }, missed_visible_class_counts: {}, high_confidence_pseudo_support_rate: 1, claim_boundary: "Pseudo-audit only.", model_requested: "fake", response_models: ["fake"] },
  frames: [{
    frame_id: "eth-office:0:000001", observation: 0, message_index: 1, timestamp_ns: 1, width: 640, height: 480,
    pose: { frame: "T_G_C", translation_m: [0, 0, 0], quaternion_xyzw: [0, 0, 0, 1] },
    image_url: "/raw.jpg", overlay_url: "/overlay.jpg", audit_status: "reviewed", missed_visible_classes: [], audit_limitations: [],
    detections: [{ detection_id: "det-1", frame_id: "eth-office:0:000001", canonical_class: "chair", phrase: "office chair", score: .91, box_xyxy: [10, 10, 100, 100], box_normalized: [.1, .1, .5, .5], mask_url: "/mask.png", mask_score: .88, mask_area_fraction: .2, warnings: [], audit_status: "supported", audit: { detection_id: "det-1", verdict: "supported", category_correct: "yes", mask_quality: "good", explanation: "Visible chair." } }],
  }],
};

afterEach(() => vi.unstubAllGlobals());

describe("ObjectsPage", () => {
  it("shows model-generated evidence, controls, and claim boundaries", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(payload), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    render(<ObjectsPage />);
    expect(await screen.findByRole("heading", { name: "What objects can the memory actually see?" })).toBeInTheDocument();
    expect(screen.getByText(/not ground truth/i)).toBeInTheDocument();
    expect(screen.getByText("384")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Visit" })).toBeInTheDocument();
    expect(screen.getByRole("slider", { name: "Minimum detector score" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "masks" }));
    expect(screen.getByAltText("Raw office frame 1")).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  });
});
