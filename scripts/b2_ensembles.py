"""Model combinations from saved predictions (folds 0 and 1), no training. The list is fixed before
any result is seen:
    a. E0 + B1;  b. E0 + E2;  c. E0 + E1 + E2 + E5;  d. c + B1;
    e. per task: P(alive) from the best of B1 / E0 / E2 on alive-vs-dead, P(home | alive) from the
       best of them on the team (frame-level, development held-out sets).
Equal weights (probability average). A combination replaces E0 only if its combined zone score
beats E0's by more than 2 paired standard errors and no source gets worse.

    python scripts/b2_ensembles.py > experiments/b2_ensembles.md
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_adopt  # noqa: E402
import b1_dev  # noqa: E402
import b2_compare  # noqa: E402
from poss import dataset, evaluate  # noqa: E402

# Params
N_SE = 2.0
COMBOS = {"a: E0+B1": ["b2e0", "b1_same"], "b: E0+E2": ["b2e0", "b2e2"],
          "c: E0+E1+E2+E5": ["b2e0", "b2e1", "b2e2", "b2e5"],
          "d: c+B1": ["b2e0", "b2e1", "b2e2", "b2e5", "b1_same"]}
TASK_CANDIDATES = ["b1_same", "b2e0", "b2e2"]


def load(tag):
    return pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))


def save(tag, pred):
    pickle.dump(pred, open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "wb"))


def average(tags, alive_from=None, team_from=None):
    preds = [load(t) for t in tags]
    out = {}
    for k in preds[0]:
        r = dict(preds[0][k])
        src_a = [load(alive_from)] if alive_from else preds
        src_t = [load(team_from)] if team_from else preds
        r["p_alive"] = np.mean([p[k]["p_alive"] for p in src_a], 0)
        r["p_home"] = np.mean([p[k]["p_home"] for p in src_t], 0)
        out[k] = r
    return out


def frame_scores(tag):
    """Alive-vs-dead accuracy (DFL simulated) and team accuracy on live frames (DFL simulated
    + SkillCorner), at 10 Hz rows, threshold 0.5."""
    acc_a, acc_t = [], []
    for k, r in load(tag).items():
        gt = r["gt"][r["near"]]
        if k.startswith("dfl_sim"):
            ok = gt < dataset.X
            acc_a.append(np.mean((r["p_alive"][ok] > 0.5) == (gt[ok] != dataset.D)))
        if k.startswith("dfl_sim") or k.startswith("sc_"):
            live = gt < dataset.D
            acc_t.append(np.mean((r["p_home"][live] > 0.5) == (gt[live] == dataset.H)))
    return np.mean(acc_a), np.mean(acc_t)


def main():
    keys = list(load("b2e0"))
    b2_compare.restricted_b1(keys, "b1_same")
    fs = {t: frame_scores(t) for t in TASK_CANDIDATES}
    best_a = max(TASK_CANDIDATES, key=lambda t: fs[t][0])
    best_t = max(TASK_CANDIDATES, key=lambda t: fs[t][1])
    print("Frame-level accuracy on the development held-out sets (alive-vs-dead on DFL simulated; "
          "team on live frames of DFL simulated + SkillCorner):\n")
    print("| model | alive/dead | team |\n|---|---|---|")
    for t in TASK_CANDIDATES:
        print(f"| {t} | {fs[t][0]:.4f} | {fs[t][1]:.4f} |")
    print(f"\nPer-task combination: P(alive) from {best_a}, P(home | alive) from {best_t}.\n")
    combos = dict(COMBOS)
    for name, tags in combos.items():
        save("ens_" + name[0], average(tags))
    save("ens_e", average(["b2e0"], alive_from=best_a, team_from=best_t))
    combos["e: per task"] = None
    w = evaluate.SC_WEIGHT
    zr, dr, cr = b1_adopt.per_source("b2e0")
    print(f"E0 reference: c_team {zr['c_team']}, DFL {zr['dfl_sim']:.3f}, SC {zr['sc_live']:.3f}, "
          f"combined {zr['combined']:.3f}\n")
    print("| combination | c_team | DFL sim | SC live | combined | diff | paired SE | adopt |")
    print("|---|---|---|---|---|---|---|---|")
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    for name in combos:
        z, d, c = b1_adopt.per_source("ens_" + name[0])
        dd = np.array([d[k] - dr[k] for k in dr])
        dc = np.array([c[k] - cr[k] for k in cr])
        diff = z["combined"] - zr["combined"]
        sec = np.sqrt(se(dd) ** 2 + (w * se(dc)) ** 2) / (1 + w)
        ok = diff > N_SE * sec and dd.mean() >= 0 and dc.mean() >= 0
        print(f"| {name} | {z['c_team']} | {z['dfl_sim']:.3f} | {z['sc_live']:.3f} | "
              f"{z['combined']:.3f} | {diff:+.3f} | {sec:.3f} | {'yes' if ok else 'no'} |",
              flush=True)


if __name__ == "__main__":
    main()
