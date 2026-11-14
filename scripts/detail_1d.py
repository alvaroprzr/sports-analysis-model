"""Why 1d did not pass: E0 trained on the control reading (b2e0c_5f) against E0 trained on the
flag (b2e0_5f), each reading, simulated and clean DFL and SkillCorner, and by duration; decoder
cost chosen excluding the evaluated fold (the same choice for every set of a fold).

    python scripts/detail_1d.py > experiments/detail_1d.md
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import xfold  # noqa: E402
from poss import decode, evaluate  # noqa: E402

MODELS = {"E0 flag-trained": "b2e0_5f", "E0 control-trained": "b2e0c_5f"}
BINS = evaluate.DURATION_BINS_S


def main():
    res, counts = {}, {}
    for name, tag in MODELS.items():
        for reading, t in (("flag", tag), ("control", xfold.with_control(tag))):
            s, chosen = xfold.scores(t)
            res[(name, reading)] = s
            pred = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{t}.pkl"), "rb"))
            acc = {}
            for k, r in pred.items():
                c = chosen[xfold.fold_of(k)]
                codes = decode.to_native(decode.decode(r["p_alive"], r["p_home"], r["period"],
                                                       (c, xfold.C_STOP, xfold.C_STOP)), r["near"], r["n"])
                gt, pr = r["gt"].copy(), codes
                if k.startswith("sc_"):
                    gt[gt == evaluate.pm.D] = evaluate.pm.X
                    pr = pr.copy()
                    pr[pr == evaluate.pm.D] = evaluate.pm.X
                src = k.split("_")[0] + ("_" + k.split("_")[1] if k.startswith("dfl") else "")
                dc = evaluate.duration_counts(gt, pr, fps=10 if k.startswith("sc_") else 25)
                acc[src] = {q: acc[src][q] + dc[q] for q in dc} if src in acc else dc
            counts[(name, reading)] = acc
    print("## Mean spellF1 (cost chosen excluding the evaluated fold)\n")
    print("| model | reading | DFL sim | DFL clean | SC live | combined (sim) |\n|---|---|---|---|---|---|")
    for (name, reading), s in res.items():
        m = lambda p: np.mean([v for k, v in s.items() if k.startswith(p)])
        print(f"| {name} | {reading} | {m('dfl_sim'):.3f} | {m('dfl_clean'):.3f} | {m('sc_'):.3f} | "
              f"{evaluate.combined(m('dfl_sim'), m('sc_')):.3f} |")
    for src, label in (("dfl_sim", "DFL simulated"), ("dfl_clean", "DFL clean"), ("sc", "SkillCorner live")):
        for reading in ("flag", "control"):
            if src == "sc" and reading == "control":
                continue
            print(f"\n## {label}, {reading} reading: F1 by true / predicted duration (recall, precision)\n")
            print("| duration | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
            tabs = {n: evaluate.duration_table(counts[(n, reading)][src]) for n in MODELS}
            for i in range(len(BINS) - 1):
                hi = f"{BINS[i + 1]:g}" if np.isfinite(BINS[i + 1]) else "inf"
                cells = [f"{tabs[n]['f1'][i]:.3f} ({tabs[n]['recall'][i]:.2f}, {tabs[n]['precision'][i]:.2f})"
                         for n in MODELS]
                print(f"| {BINS[i]:g}-{hi} s (n {int(tabs[list(MODELS)[0]]['n_gt'][i])}) | " + " | ".join(cells) + " |")
    print("\nAdoption rule (experiments/compare_1d.md): a. combined control +0.009 (SE 0.009) fails; "
          "b. mean of the two readings on simulated DFL 0.647 vs 0.690 fails; c. SkillCorner +0.009 "
          "(SE 0.003) passes.")


if __name__ == "__main__":
    main()
