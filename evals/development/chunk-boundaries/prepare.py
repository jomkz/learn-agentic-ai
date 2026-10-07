"""Copy inspected baseline cases into a separate development suite; never edit the baseline."""

import argparse
import json
import shutil
from pathlib import Path

from mobility_ai.evals.benchmark import load_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = Path(__file__).resolve().parents[2] / "benchmarks/project-docs-v1"
    # Namespace truncation, chunk defaults, missing-database recovery, and two unknowns.
    selected = {"q18", "q02", "q08", "q21", "q23"}
    questions = [row for row in load_suite(baseline) if row["id"] in selected]
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(baseline / "corpus", args.output / "corpus")
    shutil.copy2(baseline / "corpus-manifest.json", args.output / "corpus-manifest.json")
    (args.output / "questions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in questions), encoding="utf-8"
    )
    print(f"Prepared {len(questions)} inspected development questions at {args.output}")


if __name__ == "__main__":
    main()
