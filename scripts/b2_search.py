"""B2 search by successive halving: 12 configurations on fold 0, the best 4 on folds 0-2, the best
2 on all 5 folds (34 fold trainings). Ranking uses b1_compare.zone_score, as B1's search does.
Folds already trained are reused, so the script can be re-run after an interruption.

    python scripts/b2_search.py --steps N > experiments/b2_search.md
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_compare  # noqa: E402
import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
from poss import b2  # noqa: E402

# Params
SPACE = {"d": [32, 48, 64], "hidden": [48, 64, 96], "lr": [1e-3, 2e-3, 4e-3],
         "win_s": [30, 60, 120], "aux": [0.5, 1.0, 2.0], "weight_decay": [1e-4, 1e-3]}
N_CONFIGS = 12
STAGES = ((N_CONFIGS, (0,)), (4, (0, 1, 2)), (2, (0, 1, 2, 3, 4)))
SEED = 0


def configs():
    rng = np.random.default_rng(SEED)
    out = [dict(b2_dev.DEFAULT)]
    while len(out) < N_CONFIGS:
        c = dict(b2_dev.DEFAULT)
        for k, v in SPACE.items():
            c[k] = v[rng.integers(len(v))]
        if c not in out:
            out.append(c)
    return out


def fold_pred(j, i, cfg, steps):
    path = os.path.join(b1_dev.CACHE, f"pred_b2s{j}_f{i}.pkl")
    if not os.path.exists(path):
        pickle.dump(b2_dev.train_fold(i, steps, cfg=cfg), open(path, "wb"))
    return pickle.load(open(path, "rb"))


def score(j, folds, cfgs, steps):
    pred = {}
    for i in folds:
        pred.update(fold_pred(j, i, cfgs[j], steps))
    pickle.dump(pred, open(os.path.join(b1_dev.CACHE, f"pred_b2s{j}.pkl"), "wb"))
    return b1_compare.zone_score(f"b2s{j}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    a = ap.parse_args()
    b2.set_threads(b2_dev.THREADS)
    cfgs = configs()
    alive = list(range(N_CONFIGS))
    for n_keep, folds in STAGES:
        alive = alive[:n_keep]
        res = {j: score(j, folds, cfgs, a.steps) for j in alive}
        print(f"\n## folds {list(folds)}\n")
        print("| # | config | c_team | DFL sim | SC live | combined |\n|---|---|---|---|---|---|")
        for j in sorted(alive, key=lambda j: -res[j]["combined"]):
            r = res[j]
            print(f"| {j} | {json.dumps({k: cfgs[j][k] for k in SPACE})} | {r['c_team']} | "
                  f"{r['dfl_sim']:.3f} | {r['sc_live']:.3f} | {r['combined']:.3f} |", flush=True)
        alive = sorted(alive, key=lambda j: -res[j]["combined"])


if __name__ == "__main__":
    main()
