import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChangesPage } from "./ChangesPage";

const payload = {
  dataset: "ETH Office fixture",
  logical_order_note: "Logical order only.",
  claim_boundary: "Pseudo-reference, not ground truth.",
  metrics: {
    observation_count: 2,
    rgb_sample_count: 48,
    pair_count: 1,
    geometric_candidate_count: 9,
    reviewed_candidate_count: 2,
    accepted_pseudo_reference_count: 1,
    verdict_counts: { supported: 1, uncertain: 1, unsupported: 0 },
  },
  method: { voxel_size_m: 0.02, primary_threshold_m: 0.05, distance_thresholds_m: [0.02, 0.05, 0.1], min_cluster_voxels: 20 },
  observations: [0, 1].map((index) => ({
    observation_id: `eth-office:${index}`,
    logical_order: index,
    frame_count: 1,
    frames: [{ message_index: index, timestamp_ns: index, image_url: `/frame-${index}.jpg` }],
    contact_sheet_url: `/sheet-${index}.jpg`,
    vlm_contact_sheet_url: `/vlm-sheet-${index}.jpg`,
  })),
  pairs: [{
    pair_id: "0-to-1",
    earlier_observation: 0,
    current_observation: 1,
    consecutive: true,
    current_only_candidate_count: 5,
    earlier_only_candidate_count: 4,
    current_only_projection_url: "/current.png",
    earlier_only_projection_url: "/earlier.png",
    changed_fraction: { "0.050": { current_only: 0.1, earlier_only: 0.2 } },
    point_to_point: {},
    reviewed_candidates: [{ candidate_id: "candidate-0", verdict: "supported", interpretation: "current_only", description: "Visible box.", confidence: "high", evidence_ids: ["current"], limitations: [], related_candidate_id: null }],
    review_limitations: [],
  }],
};

afterEach(() => vi.unstubAllGlobals());

describe("ChangesPage", () => {
  it("shows scans, geometry, and the pseudo-reference boundary", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(payload), { status: 200 })));
    render(<ChangesPage />);
    await waitFor(() => expect(screen.getByText("Pseudo-reference, not ground truth.", { exact: false })).toBeInTheDocument());
    expect(screen.getByText("Raw geometric clusters")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "0-to-1 | consecutive" })).toBeInTheDocument();
    expect(screen.getByText("Visible box.")).toBeInTheDocument();
    expect(screen.getByAltText("Contact sheet for observation 0")).toBeInTheDocument();
  });
});
