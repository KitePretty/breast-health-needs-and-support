"""Compare recomputed aggregate tables with the archived statistical results."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    reference = ROOT / "reference_results/tables"
    actual = ROOT / "outputs/tables"
    records = []
    failures = []
    for ref_path in sorted(reference.glob("*.csv")):
        new_path = actual / ref_path.name
        record = {"table": ref_path.name}
        try:
            ref, new = pd.read_csv(ref_path), pd.read_csv(new_path)
            assert list(ref.columns) == list(new.columns), "column mismatch"
            assert ref.shape == new.shape, "row count mismatch"
            max_difference = 0.0
            for col in ref:
                if pd.api.types.is_numeric_dtype(ref[col]) and pd.api.types.is_numeric_dtype(new[col]):
                    np.testing.assert_allclose(new[col], ref[col], rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=col)
                    differences = (new[col] - ref[col]).abs().dropna()
                    if len(differences):
                        max_difference = max(max_difference, float(differences.max()))
                else:
                    assert new[col].fillna("<NA>").astype(str).equals(ref[col].fillna("<NA>").astype(str)), f"value mismatch in {col}"
            record.update(status="passed", rows=len(ref), max_absolute_numeric_difference=max_difference)
        except (AssertionError, FileNotFoundError) as error:
            record.update(status="failed", detail=str(error))
            failures.append(ref_path.name)
        records.append(record)
    assert len(records) == 24, "Expected 24 archived aggregate result tables"
    result = {"tables_checked": len(records), "passed": len(records) - len(failures),
              "failed": len(failures), "relative_tolerance": 1e-8, "absolute_tolerance": 1e-10,
              "comparisons": records}
    output = ROOT / "outputs/logs"
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
    if failures:
        raise SystemExit("Result comparison failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
