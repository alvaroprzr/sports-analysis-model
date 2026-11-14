"""Checks that our scoring wrappers (src/poss/evaluate.py) agree with the official
evaluation/possession_metrics.py, on synthetic cases with known answers and on random sequences.

    python scripts/test_evaluate.py > experiments/test_evaluate.md
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from poss import evaluate  # noqa: E402
from poss.evaluate import pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X
results = []


def seq(*parts):
    return np.concatenate([np.full(n, c, np.int8) for c, n in parts])


def check(name, got, want, tol=1e-12):
    ok = abs(got - want) <= tol
    results.append(ok)
    print(f"| {name} | {got:.4f} | {want:.4f} | {'ok' if ok else 'FAIL'} |")


def main():
    print("| case | ours | expected | |\n|---|---|---|---|")
    gt = seq((X, 50), (H, 100), (A, 100), (D, 50), (H, 100), (X, 50))
    check("perfect prediction", evaluate.full_scores(gt, gt)["spellF1"], 1.0)
    # an X inside a true possession is filled with the previous answer: no penalty
    p = gt.copy()
    p[100:120] = X
    check("X inside a possession (filled with previous label)",
          evaluate.full_scores(gt, p)["spellF1"], 1.0)
    # labels outside the graded runs (gt X) are ignored
    p = gt.copy()
    p[:50] = A
    p[-50:] = H
    check("labels on ungraded frames ignored", evaluate.full_scores(gt, p)["spellF1"], 1.0)
    # one spell shifted so that its IoU with the truth is exactly 0.5 (100 frames, overlap 2/3):
    # 3 spells; the shifted H matches at IoU <= 0.5 only (3 of 5 thresholds)
    gt2 = seq((H, 90), (A, 90), (D, 30), (H, 90))
    p2 = seq((H, 60), (A, 120), (D, 30), (H, 90))
    ours = evaluate.full_scores(gt2, p2)["spellF1"]
    check("partial overlap equals the official scorer", ours,
          pm.evaluate(gt2, p2)["spell_f1_mean"])
    # one-to-one matching: two predicted spells cannot both match one true spell
    gt3 = seq((H, 200), (D, 20), (A, 100))
    p3 = seq((H, 100), (A, 2), (H, 98), (D, 20), (A, 100))
    check("one-to-one matching equals the official scorer", evaluate.full_scores(gt3, p3)["spellF1"],
          pm.evaluate(gt3, p3)["spell_f1_mean"])
    # random sequences
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = rng.integers(5, 40)
        g = np.repeat(rng.choice([H, A, D, X], n, p=[.35, .35, .2, .1]), rng.integers(5, 300, n))
        q = g.copy()
        flip = rng.random(len(q)) < 0.01
        q[flip] = rng.choice([H, A, D, X], flip.sum())
        q = np.repeat(q[::7], 7)[:len(g)]
        worst = max(worst, abs(evaluate.full_scores(g, q)["spellF1"] -
                               pm.evaluate(g, q)["spell_f1_mean"]))
    check("200 random sequences: max |ours - official|", worst, 0.0)
    # live score (our convention for SkillCorner): D becomes X in both, so only H / A spells count
    gt4 = seq((H, 100), (D, 50), (A, 100))
    p4 = seq((H, 100), (H, 50), (A, 100))
    check("live score ignores dead-ball frames", evaluate.live_scores(gt4, p4)["spellF1_live"], 1.0)
    print(f"\n{sum(results)} of {len(results)} checks pass. Graded-half selection on BAR-ATH: "
          "done by the organisers' scorer from the ground truth (frames outside the halves are X "
          "there); on our development matches frames outside play are X in our ground truth, so "
          "the same official functions skip them.")


if __name__ == "__main__":
    main()
