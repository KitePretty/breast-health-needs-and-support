"""Validate the coded input contract and response/missing-content distinctions."""
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    with (DATA / "data_dictionary.csv").open(newline="", encoding="utf-8") as f:
        dictionary = list(csv.DictReader(f))
    expected = {name: {r["field"] for r in dictionary if r["file"] == name}
                for name in ["threads.csv", "comments.csv"]}
    frames = {}
    for name in expected:
        with (DATA / name).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == expected[name], f"Unexpected or missing columns in {name}"
            frames[name] = list(reader)
    t, c = frames["threads.csv"], frames["comments.csv"]
    assert len(t) == 1000 and len(c) == 2532
    tids = [r["thread_id"] for r in t]
    cids = [r["comment_id"] for r in c]
    assert len(set(tids)) == len(tids) and len(set(cids)) == len(cids)
    assert all(re.fullmatch(r"T_[0-9a-f]{24}", v) for v in tids)
    assert all(re.fullmatch(r"C_[0-9a-f]{24}", v) for v in cids)
    assert {r["thread_id"] for r in c} <= set(tids)
    with (DATA / "topics.csv").open(newline="", encoding="utf-8") as f:
        topics = {r["topic_id"]: r["topic_label"] for r in csv.DictReader(f)}
    post_codes = [r["field"] for r in dictionary
                  if r["file"] == "threads.csv" and r["field"].startswith(("A", "B", "D"))]
    for r in t:
        assert r["topic_label"] == topics[r["topic_id"]]
        assert 1 <= int(r["rank"]) <= 100 and 2022 <= int(r["year"]) <= 2025
        assert r["C1_interval"] in {"", "1.0", "2.0", "3.0", "4.0", "1", "2", "3", "4"}
        for field in post_codes:
            allowed = {"0", "1", "0.0", "1.0", ""} if field.startswith("D2_") else {"0", "1", "0.0", "1.0"}
            assert r[field] in allowed, (field, "unexpected code")
        assert r["C3_has_human_response"] in {"0", "1"}
        assert r["actual_analyzable_comment"] in {"0", "1"}
    topic_ranks = [(r["topic_id"], int(r["rank"])) for r in t]
    assert len(set(topic_ranks)) == 1000
    counts = {}
    for r in c:
        counts[r["thread_id"]] = counts.get(r["thread_id"], 0) + 1
        assert all(v in {"0", "1"} for k, v in r.items() if k.startswith("E"))
    for r in t:
        n = counts.get(r["thread_id"], 0)
        assert int(r["n_analyzable_comments"]) == n
        assert int(r["actual_analyzable_comment"]) == int(n > 0)
        assert n == 0 or r["C3_has_human_response"] == "1"
    primary = [r for r in t if r["C1_interval"] != ""]
    report = [r for r in primary if r["topic_label"] == "Reading Clinical Reports"]
    result = {
        "threads": len(t), "comments": len(c), "primary_threads": len(primary),
        "human_response_threads": sum(int(r["C3_has_human_response"]) for r in t),
        "analyzable_response_threads": len(counts),
        "response_content_unknown_threads": sum(r["C3_has_human_response"] == "1" and r["actual_analyzable_comment"] == "0" for r in t),
        "primary_human_response_threads": sum(int(r["C3_has_human_response"]) for r in primary),
        "clinical_report_threads": len(report),
        "clinical_report_human_response_threads": sum(int(r["C3_has_human_response"]) for r in report),
        "D2_blank_all_threads": sum(r["D2_risk_assessment"] == "" for r in t),
        "D2_blank_primary_threads": sum(r["D2_risk_assessment"] == "" for r in primary),
        "status": "passed",
    }
    assert (result["primary_threads"], result["human_response_threads"], result["analyzable_response_threads"],
            result["response_content_unknown_threads"], result["primary_human_response_threads"]) == (957, 730, 719, 11, 698)
    assert (result["clinical_report_threads"], result["clinical_report_human_response_threads"]) == (100, 32)
    output = ROOT / "outputs/logs"
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
