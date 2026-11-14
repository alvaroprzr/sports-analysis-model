"""B1 arm "SkillCorner everywhere": like b1_dev.py's sim arm, but SkillCorner also trains the
alive / dead model with its approximate dead-ball label (gaps between its phases of play), with
the same half total weight. Compared with the sim arm (SkillCorner only for the team in live play)
and sim_nosc (no SkillCorner) to decide how SkillCorner enters training.

    python scripts/b1_scfull.py   # -> data/cache/b1/pred_sim_scfull.pkl
"""

import os
import pickle
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
from poss import dataset  # noqa: E402


def train_fold(i, params=b1_dev.GBM):
    test_dfl, test_sc = b1_dev.folds()[i]
    train_dfl = [m for m in b1_dev.SPLIT["dfl"]["dev"] if m != test_dfl]
    train_sc = [m for m in b1_dev.SPLIT["skillcorner"]["dev"] if m not in test_sc]
    step = b1_dev.ROW_STEP
    pieces_d, pieces_s = [], []
    for mid in train_dfl:
        for seed in range(b1_dev.N_DEGRADE):
            for swap in (False, True):
                F, _, y, _, _ = b1_dev.dfl_rows(mid, seed, swap)
                F, y = F[::step], y[::step]
                k = y < dataset.X
                pieces_d.append((F[k], y[k]))
                del F
    for mid in train_sc:
        for swap in (False, True):
            F, _, y, _, _ = b1_dev.sc_rows(mid, swap)
            F, y = F[::step], y[::step]
            k = y < dataset.X
            pieces_s.append((F[k], y[k]))
            del F
    n_cols = pieces_d[0][0].shape[1]
    nd = sum(len(y) for _, y in pieces_d)
    X, yall = b1_dev._stack(pieces_d + pieces_s, n_cols)
    del pieces_d, pieces_s
    ns = len(yall) - nd
    w_alive = np.r_[np.ones(nd), np.full(ns, b1_dev.SC_TOTAL_WEIGHT * nd / max(ns, 1))]
    alive = lgb.LGBMClassifier(**params).fit(X, (yall != dataset.D).astype(int),
                                             sample_weight=w_alive)
    live = yall < dataset.D
    src_d = np.r_[np.ones(nd, bool), np.zeros(ns, bool)][live]
    X, yall = X[live], yall[live]
    ld = int(src_d.sum())
    wt = np.where(src_d, 1.0, b1_dev.SC_TOTAL_WEIGHT * ld / max(len(yall) - ld, 1))
    team = lgb.LGBMClassifier(**params).fit(X, (yall == dataset.H).astype(int), sample_weight=wt)
    del X
    out = {}
    tests = [(f"dfl_sim_{test_dfl}", b1_dev.dfl_rows(test_dfl, b1_dev.TEST_SEED, False),
              b1_dev.load("dfl", test_dfl)),
             (f"dfl_clean_{test_dfl}", b1_dev.dfl_rows(test_dfl, 0, False, clean=True),
              b1_dev.load("dfl", test_dfl))]
    tests += [(f"sc_{s}", b1_dev.sc_rows(s, False), b1_dev.load("skillcorner", s))
              for s in test_sc]
    for name, (F, _, y, period, near), m in tests:
        out[name] = dict(p_alive=alive.predict_proba(F)[:, 1], p_home=team.predict_proba(F)[:, 1],
                         period=period, near=near,
                         gt=dataset.codes(m.alive, m.team, m.period), n=m.n_frames)
    return out


def main():
    pred = {}
    for i in range(b1_dev.N_FOLDS):
        pred.update(train_fold(i))
        print("fold", i, "done", flush=True)
    pickle.dump(pred, open(os.path.join(b1_dev.CACHE, "pred_sim_scfull.pkl"), "wb"))


if __name__ == "__main__":
    main()
