#!/usr/bin/env python3
"""Run the ai_autoboost audit/experiment rounds in order.

This wrapper intentionally runs the non-destructive ai_autoboost scripts. It
does not force re-downloads or overwrite source manuscript/data files.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


SCRIPTS = [
    ROOT / "ai_autoboost" / "code" / "round0_audit" / "round0_audit.py",
    ROOT / "ai_autoboost" / "code" / "round1_reproducibility" / "round1_reproducibility.py",
    ROOT / "ai_autoboost" / "code" / "round2_baselines_ablation" / "round2_baselines_ablation.py",
    ROOT / "ai_autoboost" / "code" / "round3_mechanism_error" / "round3_mechanism_error.py",
    ROOT / "ai_autoboost" / "code" / "round4_generalization_final" / "round4_generalization_final.py",
]

ROUND5_REPLICATES = ROOT / "ai_autoboost" / "code" / "round5_literature_counterfactual" / "round5_counterfactual_replicates.py"
ROUND5_SUMMARY = ROOT / "ai_autoboost" / "code" / "round5_literature_counterfactual" / "round5_summarize_counterfactual.py"
ROUND6_ASSETS = ROOT / "ai_autoboost" / "code" / "round6_submission_assets" / "round6_finalize_submission_assets.py"
ROUND7_FULL_MANUSCRIPT = ROOT / "ai_autoboost" / "code" / "round7_full_manuscript" / "round7_build_full_article.py"


def main() -> None:
    for script in SCRIPTS:
        print(f"\n=== Running {script.relative_to(ROOT)} ===")
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print(f"\n=== Running {ROUND5_REPLICATES.relative_to(ROOT)} ===")
    subprocess.run([sys.executable, str(ROUND5_REPLICATES), "--reps", "5000"], cwd=ROOT, check=True)
    print(f"\n=== Running {ROUND5_SUMMARY.relative_to(ROOT)} ===")
    subprocess.run([sys.executable, str(ROUND5_SUMMARY)], cwd=ROOT, check=True)
    print(f"\n=== Running {ROUND6_ASSETS.relative_to(ROOT)} ===")
    subprocess.run([sys.executable, str(ROUND6_ASSETS)], cwd=ROOT, check=True)
    print(f"\n=== Running {ROUND7_FULL_MANUSCRIPT.relative_to(ROOT)} ===")
    subprocess.run([sys.executable, str(ROUND7_FULL_MANUSCRIPT)], cwd=ROOT, check=True)
    print("\nAll ai_autoboost rounds completed.")


if __name__ == "__main__":
    main()
