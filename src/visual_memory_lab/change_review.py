"""Structured VLM review of ETH Office geometric change candidates."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

PROMPT_VERSION = "phase6a-change-review-v1"
DEFAULT_MODEL = "gpt-5.6-terra"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateReview(StrictModel):
    candidate_id: str
    verdict: Literal["supported", "unsupported", "uncertain"]
    interpretation: Literal["current_only", "earlier_only", "possible_move", "unknown"]
    description: str
    confidence: Literal["low", "medium", "high"]
    evidence_ids: list[str]
    limitations: list[str]
    related_candidate_id: str | None


class PairReview(StrictModel):
    pair_id: str
    candidates: list[CandidateReview]
    overall_limitations: list[str]


def _image_part(path: Path) -> tuple[dict[str, str], str]:
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return (
        {
            "type": "input_image",
            "image_url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
            "detail": "high",
        },
        hashlib.sha256(data).hexdigest(),
    )


def _cache_path(
    *, cache_dir: Path, model: str, prompt: str, image_hashes: list[str]
) -> Path:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "schema": PairReview.model_json_schema(),
            "prompt": prompt,
            "image_hashes": image_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return cache_dir / f"{hashlib.sha256(payload).hexdigest()}.json"


def _validate_review(
    review: PairReview,
    *,
    pair_id: str,
    candidate_ids: list[str],
    evidence_ids: list[str],
) -> None:
    if review.pair_id != pair_id:
        raise ValueError(f"review pair ID {review.pair_id!r} does not match {pair_id!r}")
    actual = [item.candidate_id for item in review.candidates]
    if len(actual) != len(set(actual)) or set(actual) != set(candidate_ids):
        raise ValueError("VLM review IDs do not exactly match requested candidate IDs")
    candidate_set = set(candidate_ids)
    evidence_set = set(evidence_ids)
    for item in review.candidates:
        unknown_evidence = sorted(set(item.evidence_ids) - evidence_set)
        if unknown_evidence:
            raise ValueError(f"candidate {item.candidate_id} cited unknown evidence: {unknown_evidence}")
        if item.related_candidate_id is not None:
            if item.related_candidate_id not in candidate_set:
                raise ValueError(f"candidate {item.candidate_id} has an unknown related candidate")
            if item.related_candidate_id == item.candidate_id:
                raise ValueError(f"candidate {item.candidate_id} cannot relate to itself")
        if item.verdict == "supported" and not item.evidence_ids:
            raise ValueError(f"supported candidate {item.candidate_id} must cite evidence")


def _call_review(
    *,
    client: object,
    model: str,
    pair_id: str,
    candidates: list[dict[str, object]],
    evidence: list[tuple[str, Path]],
    cache_dir: Path,
) -> tuple[PairReview, str, bool]:
    candidate_lines = "\n".join(
        f"- {item['candidate_id']}: direction={item['direction']}, "
        f"centroid_m={item['centroid_m']}, voxel_count={item['voxel_count']}"
        for item in candidates
    )
    evidence_ids = [item[0] for item in evidence]
    prompt = (
        "You are reviewing public ETH Office change-detection evidence. The numbered colored "
        "clusters were produced by geometric mesh differencing. Review every supplied candidate "
        "exactly once. A cluster may be a physical scene difference, reconstruction noise, missing "
        "coverage, or uncertain. Use supported only when the RGB contact sheets or the paired 3D "
        "projections visibly support it. Do not claim human activity, cause, calendar time, exhaustive "
        "recall, or ground truth. possible_move requires a plausible current-only/earlier-only pair; "
        "cite the related candidate ID. Cite only the supplied evidence IDs.\n\n"
        f"Pair: {pair_id}\nEvidence IDs: {', '.join(evidence_ids)}\nCandidates:\n{candidate_lines}"
    )
    content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
    image_hashes: list[str] = []
    for evidence_id, path in evidence:
        part, digest = _image_part(path)
        content.append({"type": "input_text", "text": f"Evidence {evidence_id}:"})
        content.append(part)
        image_hashes.append(digest)
    cache_path = _cache_path(
        cache_dir=cache_dir, model=model, prompt=prompt, image_hashes=image_hashes
    )
    candidate_ids = [str(item["candidate_id"]) for item in candidates]
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        review = PairReview.model_validate(cached["parsed"])
        _validate_review(
            review,
            pair_id=pair_id,
            candidate_ids=candidate_ids,
            evidence_ids=evidence_ids,
        )
        return review, str(cached["response_model"]), True

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = getattr(client, "responses").parse(
                model=model,
                input=[{"role": "user", "content": content}],
                text_format=PairReview,
                store=False,
            )
            review = response.output_parsed
            if not isinstance(review, PairReview):
                raise ValueError("VLM response did not match the required review schema")
            _validate_review(
                review,
                pair_id=pair_id,
                candidate_ids=candidate_ids,
                evidence_ids=evidence_ids,
            )
            response_model = str(getattr(response, "model", model))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "prompt_version": PROMPT_VERSION,
                        "model_requested": model,
                        "response_model": response_model,
                        "image_hashes": image_hashes,
                        "parsed": review.model_dump(mode="json"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return review, response_model, False
        except Exception as error:  # SDK and transport exceptions vary.
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    assert last_error is not None
    raise RuntimeError(f"VLM review failed after three attempts: {last_error}") from last_error


def review_eth_changes(
    *,
    baseline: Path,
    audit: Path,
    output: Path,
    cache_dir: Path,
    model: str = DEFAULT_MODEL,
    client: object | None = None,
) -> dict[str, object]:
    """Review the largest geometric candidates and freeze a pseudo-reference."""

    load_dotenv()
    resolved_output = output.resolve()
    if resolved_output.exists() and (not resolved_output.is_dir() or any(resolved_output.iterdir())):
        raise FileExistsError(f"output path is not empty: {resolved_output}")
    resolved_output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((audit / "manifest.json").read_text(encoding="utf-8"))
    pair_records = [json.loads(line) for line in (baseline / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    candidates = [json.loads(line) for line in (baseline / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line]
    observations = {
        int(item["logical_order"]): item for item in manifest["observations"]
    }
    if client is None:
        from openai import OpenAI

        client = OpenAI()

    reviews: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    for pair in pair_records:
        pair_id = str(pair["pair_id"])
        pair_candidates = [item for item in candidates if item["pair_id"] == pair_id]
        selected: list[dict[str, object]] = []
        for direction in ("current-only", "earlier-only"):
            directional = [item for item in pair_candidates if item["direction"] == direction]
            directional.sort(key=lambda item: (-int(item["voxel_count"]), str(item["candidate_id"])))
            selected.extend(directional[:6])
        if not selected:
            continue
        earlier_index = int(pair["earlier_observation"])
        current_index = int(pair["current_observation"])
        evidence = [
            (f"eth-office:{earlier_index}:rgb-contact-sheet", Path(observations[earlier_index]["bag"]["vlm_contact_sheet"])),
            (f"eth-office:{current_index}:rgb-contact-sheet", Path(observations[current_index]["bag"]["vlm_contact_sheet"])),
            (f"eth-office:{pair_id}:current-only-projection", Path(pair["evidence"]["current_only_png"])),
            (f"eth-office:{pair_id}:earlier-only-projection", Path(pair["evidence"]["earlier_only_png"])),
        ]
        review, response_model, cached = _call_review(
            client=client,
            model=model,
            pair_id=pair_id,
            candidates=selected,
            evidence=evidence,
            cache_dir=cache_dir.resolve(),
        )
        payload = {
            **review.model_dump(mode="json"),
            "model": response_model,
            "cached": cached,
            "reviewed_candidate_count": len(selected),
            "evidence": {evidence_id: str(path.resolve()) for evidence_id, path in evidence},
        }
        reviews.append(payload)
        candidate_by_id = {str(item["candidate_id"]): item for item in selected}
        for item in review.candidates:
            if item.verdict == "supported" and item.confidence in {"medium", "high"}:
                accepted.append(
                    {
                        **candidate_by_id[item.candidate_id],
                        "review": item.model_dump(mode="json"),
                        "reference_status": "vlm-supported-pseudo-reference",
                    }
                )

    counts = {verdict: 0 for verdict in ("supported", "unsupported", "uncertain")}
    total = 0
    for review in reviews:
        for item in review["candidates"]:
            counts[str(item["verdict"])] += 1
            total += 1
    summary: dict[str, object] = {
        "schema_version": 1,
        "model_requested": model,
        "prompt_version": PROMPT_VERSION,
        "pair_review_count": len(reviews),
        "reviewed_candidate_count": total,
        "verdict_counts": counts,
        "accepted_pseudo_reference_count": len(accepted),
        "claim_boundary": "VLM-supported candidates are a pseudo-reference, not human ground truth.",
    }
    (resolved_output / "reviews.json").write_text(json.dumps({"summary": summary, "pairs": reviews}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (resolved_output / "pseudo_reference.json").write_text(json.dumps({"summary": summary, "candidates": accepted}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_review_report(summary, reviews, resolved_output)
    return summary


def _write_review_report(
    summary: dict[str, object], reviews: list[dict[str, object]], output: Path
) -> None:
    sections: list[str] = []
    for review in reviews:
        rows = []
        for candidate in review["candidates"]:
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(str(candidate['candidate_id']))}</code></td>"
                f"<td>{html.escape(str(candidate['verdict']))}</td>"
                f"<td>{html.escape(str(candidate['confidence']))}</td>"
                f"<td>{html.escape(str(candidate['interpretation']))}</td>"
                f"<td>{html.escape(str(candidate['description']))}</td>"
                "</tr>"
            )
        evidence_links = []
        for evidence_id, path_value in review["evidence"].items():
            relative = Path(os.path.relpath(path_value, output)).as_posix()
            evidence_links.append(
                f'<a href="{html.escape(relative)}">{html.escape(str(evidence_id))}</a>'
            )
        sections.append(
            f"<section><h2>{html.escape(str(review['pair_id']))}</h2>"
            f"<p>{' | '.join(evidence_links)}</p>"
            "<table><thead><tr><th>Candidate</th><th>Verdict</th><th>Confidence</th>"
            f"<th>Interpretation</th><th>Description</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
        )
    counts = summary["verdict_counts"]
    assert isinstance(counts, dict)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ETH Office VLM pseudo-reference</title><style>
body{{font:15px/1.5 system-ui;margin:0 auto;max-width:1500px;padding:32px;background:#f5f2e9;color:#20231f}}
h1,h2{{font-family:Georgia,serif}} .warning{{padding:14px 18px;background:#f7dfd9;border-left:4px solid #a84032}}
table{{border-collapse:collapse;width:100%;background:white}} th,td{{padding:8px;border:1px solid #ddd;text-align:left;vertical-align:top}}
code{{font-size:12px}} section{{margin:36px 0;overflow-x:auto}}
</style></head><body><h1>ETH Office VLM pseudo-reference</h1>
<p class="warning"><strong>Claim boundary:</strong> VLM-supported candidates are not human ground truth.</p>
<p>{summary['reviewed_candidate_count']} reviewed | {counts['supported']} supported | {counts['uncertain']} uncertain |
{counts['unsupported']} unsupported | {summary['accepted_pseudo_reference_count']} medium/high-confidence accepted.</p>
{"".join(sections)}</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
