import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { HomePage } from "./HomePage";

const capabilities = { dataset: "7-scenes-office", memory_count: 6000, query_count: 4000, captured_at_available: false, analysis_available: true, analysis_requires_confirmation: true, search_modes: ["text", "image"], supported_question_families: ["location"], unsupported_claims: ["calendar-time"] };
const search = { query: { kind: "text", question: "Where is the desk?" }, display_k: 5, temporal: { captured_at: null, message: "Calendar time is unavailable for this public dataset." }, likely_area: { slug: "window-desk", name: "Window desk", support_count: 7, evidence_count: 10, strength: "strong" }, evidence: Array.from({ length: 10 }, (_, index) => ({ rank: index + 1, score: .95 - index / 100, observation_id: `memory:${index}`, collection: "memory", sequence_id: "seq-01", frame: index, captured_at: null, zone: { slug: "window-desk", name: "Window desk" }, image_url: `/api/images/memory/memory:${index}` })) };

afterEach(() => vi.unstubAllGlobals());

describe("HomePage", () => {
  it("retrieves evidence locally and requires confirmation before analysis", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/capabilities")) return new Response(JSON.stringify(capabilities), { status: 200 });
      if (url.endsWith("/api/search/text")) return new Response(JSON.stringify(search), { status: 200 });
      throw new Error(`unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<MemoryRouter><HomePage /></MemoryRouter>);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Find evidence" }));
    expect(await screen.findByText("Window desk", { selector: "h2" })).toBeInTheDocument();
    expect(screen.getAllByAltText(/Office memory/)).toHaveLength(5);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    fireEvent.click(screen.getByRole("button", { name: "Analyze selected evidence" }));
    expect(screen.getByRole("dialog", { name: "Confirm cloud analysis" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
