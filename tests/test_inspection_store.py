from pathlib import Path

from visual_memory_lab.inspection_store import InspectionStore


def test_inspection_store_create_list_and_reopen(tmp_path: Path) -> None:
    store = InspectionStore(tmp_path / "inspections.sqlite3")
    created = store.create(
        title="Desk check",
        question="Where was this desk seen before?",
        result_text="Evidence saved.",
        status="supported_with_limits",
        limitations=["No identity claim."],
        current_image_path=None,
        evidence=[{"observation_id": "office:seq-01:000001", "rank": 1, "role": "supporting"}],
    )
    assert created["question"] == "Where was this desk seen before?"
    assert len(store.list()) == 1
    reopened = store.get(str(created["id"]))
    assert reopened["evidence"][0]["observation_id"] == "office:seq-01:000001"

    with_summary = store.set_summary(str(created["id"]), {"summary": "A desk", "visible_objects": ["desk"]})
    assert with_summary["summary_json"]["summary"] == "A desk"
    with_report = store.set_report(
        str(created["id"]),
        {"status": "observed", "summary": "Desk visible"},
        result_text="Desk visible",
        status="observed",
        limitations=[],
    )
    assert with_report["report_json"]["status"] == "observed"
