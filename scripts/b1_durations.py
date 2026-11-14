"""spellF1-style recall / precision / F1 by possession duration for cached predictions.

Recall is binned by the true possession's duration, precision by the predicted one's, both with
the official greedy matching and averaged over IoU 0.3-0.7 (evaluate.duration_counts).

    python scripts/b1_durations.py TAG COSTS     e.g.  sim 2.5,6,6
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
from poss import decode, evaluate  # noqa: E402

pm = evaluate.pm


def live(c):
    c = c.copy()
    c[c == pm.D] = pm.X
    return c


def main():
    tag = sys.argv[1]
    costs = tuple(float(x) for x in sys.argv[2].split(","))
    pred = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))
    sums = {}
    for name, r in pred.items():
        c = decode.to_native(decode.decode(r["p_alive"], r["p_home"], r["period"], costs),
                             r["near"], r["n"])
        group = "DFL simulated" if "_sim_" in name else "DFL clean" if name.startswith("dfl") \
            else "SkillCorner live"
        gt, pr = (live(r["gt"]), live(c)) if group == "SkillCorner live" else (r["gt"], c)
        fps = 10 if group == "SkillCorner live" else 25
        cnt = evaluate.duration_counts(gt, pr, fps=fps)
        sums[group] = cnt if group not in sums else {k: sums[group][k] + v for k, v in cnt.items()}
    edges = evaluate.DURATION_BINS_S
    labels = [f"{a:g}-{b:g} s" if np.isfinite(b) else f">{a:g} s" for a, b in zip(edges, edges[1:])]
    print(f"{tag} at costs {costs}\n")
    for group, cnt in sums.items():
        t = evaluate.duration_table(cnt)
        print(f"### {group}\n\n| duration | true possessions | recall | predicted | precision | F1 |")
        print("|---|---|---|---|---|---|")
        for k, lab in enumerate(labels):
            print(f"| {lab} | {int(t['n_gt'][k])} | {t['recall'][k]:.3f} | {int(t['n_pred'][k])} | "
                  f"{t['precision'][k]:.3f} | {t['f1'][k]:.3f} |")
        print()


if __name__ == "__main__":
    main()
