"""Hyperparameter search for B1 with a fixed budget (the same number of configurations B2 gets).

Configurations are drawn at random (fixed seed) from a small space; the first is the reference
setting. Each runs the 5 development folds of b1_dev.py (sim arm) and is scored with
b1_compare.zone_score (the team-switch cost where both sources are nearest their own best), per
source and combined. B2's search uses the same score.

    python scripts/b1_search.py N_CONFIGS > experiments/b1_search.md
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_compare  # noqa: E402
import b1_dev  # noqa: E402

# Params
SPACE = {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1],
         "min_child_samples": [50, 100, 300], "colsample_bytree": [0.3, 0.5, 0.8]}
SEED = 0


def configs(n):
    rng = np.random.default_rng(SEED)
    out = [dict(b1_dev.GBM)]
    while len(out) < n:
        c = dict(b1_dev.GBM)
        for k, v in SPACE.items():
            c[k] = v[rng.integers(len(v))]
        c["n_estimators"] = int(round(400 * 0.05 / c["learning_rate"]))  # same learning budget
        if c not in out:
            out.append(c)
    return out


def main():
    n = int(sys.argv[1])
    print("| # | num_leaves | lr | min_child | colsample | c_team | DFL sim | SC live | combined |")
    print("|---|---|---|---|---|---|---|---|---|")
    for j, params in enumerate(configs(n)):
        path = os.path.join(b1_dev.CACHE, f"pred_search{j}.pkl")
        if not os.path.exists(path):
            pred = {}
            for i in range(b1_dev.N_FOLDS):
                pred.update(b1_dev.train_fold(i, "sim", params))
            pickle.dump(pred, open(path, "wb"))
        z = b1_compare.zone_score(f"search{j}")
        print(f"| {j} | {params['num_leaves']} | {params['learning_rate']} | "
              f"{params['min_child_samples']} | {params['colsample_bytree']} | {z['c_team']} | "
              f"{z['dfl_sim']:.3f} | {z['sc_live']:.3f} | {z['combined']:.3f} |", flush=True)


if __name__ == "__main__":
    main()
