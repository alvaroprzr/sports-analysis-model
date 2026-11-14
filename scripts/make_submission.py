"""Regenerate the submission file from the data bundle and the open external data.

Every parameter comes from configs/b1_ext.json and was chosen on the external development matches
(scripts/b1_dev.py); nothing is tuned on BAR-ATH. The pipeline is CPU-only and deterministic:

    1. calibrate the tracking simulator on external SkillCorner tracking (sim_external.py);
    2. build training rows from the external matches (DFL degraded as configured, SkillCorner);
    3. fit P(alive) and P(home | alive) (LightGBM, fixed seeds and threads);
    4. compute the physics rows of BAR-ATH, predict, decode, write <team>.csv.

    python scripts/make_submission.py --team NAME [--out DIR] [--cache DIR]

--cache keeps intermediate files (calibration, rows) between runs; a clean run is the reference
for reproducibility (compare the printed SHA-256 of two runs).
"""

import argparse
import hashlib
import json
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402
from poss import published, barath, dataset, decode, sim_external, simulate  # noqa: E402
from poss.match import Match  # noqa: E402

LABELS = np.array(["H", "A", "D", "X"])
ROUND = 4  # decimals kept of each probability before decoding


def load(src, mid):
    return Match.load(os.path.join(ROOT, "data", "external", f"{src}_v3", f"{mid}.npz"))


def cached(cache, name, make):
    if cache is None:
        return make()
    path = os.path.join(cache, name + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = make()
    os.makedirs(cache, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    return out


def copy_cal(cfg, cal, mid, seed):
    """Simulator of one DFL training copy: the calibrated one, or with domain randomisation a
    per-copy intensity (sim_external.dr_copy, the same draw as in development)."""
    if not cfg.get("domain_randomisation"):
        return cal
    dr = cfg["domain_randomisation"]
    return sim_external.dr_copy(cal, dr["table"], mid, seed, tuple(dr["vis"]), tuple(dr["tracks"]))


def dfl_pieces(cfg, cal, cache):
    """(features, labels) of every DFL training copy, one at a time."""
    step = cfg["row_step"]
    for mid in cfg["dfl_train"]:
        seeds = range(cfg["n_degrade"]) if cfg["arm"] == "sim" else [0]
        for seed in seeds:
            for swap in (False, True):
                def make():
                    m = load("dfl", mid)
                    if cfg["arm"] == "sim":
                        m = simulate.degrade(m, copy_cal(cfg, cal, mid, seed), np.random.default_rng(seed),
                                             spread=cfg["spread"])
                    elif cfg["arm"] == "filtered":
                        m = simulate.degrade(m, cal, np.random.default_rng(seed),
                                             parts=("smooth",))
                    m = dataset.swap_teams(m) if swap else m
                    F, _, y, _, _ = dataset.rows(m)
                    return F[::step], y[::step]
                yield cached(cache, f"dfl_{mid}_{cfg['arm']}{seed}_{int(swap)}", make)


def sc_pieces(cfg, cache):
    step = cfg["row_step"]
    for mid in cfg["sc_train"]:
        for swap in (False, True):
            def make():
                m = load("skillcorner", mid)
                m = dataset.swap_teams(m) if swap else m
                F, _, y, _, _ = dataset.rows(m)
                return F[::step], y[::step]
            yield cached(cache, f"sc_{mid}_{int(swap)}", make)


def fit(cfg, cal, cache):
    """P(alive) on DFL, then P(home | alive) on DFL + SkillCorner live rows.

    Arrays are built piece by piece and released as soon as they are no longer needed: the full
    training set is several GB and must fit next to LightGBM in a few more."""
    params = cfg["gbm"]
    pieces = [(F[y < dataset.X], y[y < dataset.X]) for F, y in dfl_pieces(cfg, cal, cache)]
    Xd = np.concatenate([p[0] for p in pieces])
    yd = np.concatenate([p[1] for p in pieces])
    del pieces
    alive = lgb.LGBMClassifier(**params).fit(Xd, (yd != dataset.D).astype(int))
    ld = yd < dataset.D
    Xd, yd = Xd[ld], yd[ld]
    pieces = [(F[y < dataset.D], y[y < dataset.D]) for F, y in sc_pieces(cfg, cache)]
    ns = sum(len(p[1]) for p in pieces)
    Xt = np.empty((len(yd) + ns, Xd.shape[1]), np.float32)
    Xt[:len(yd)] = Xd
    del Xd
    yt = [yd]
    k = len(yd)
    for F, y in pieces:
        Xt[k:k + len(y)] = F
        yt.append(y)
        k += len(y)
    del pieces
    yt = (np.concatenate(yt) == dataset.H).astype(int)
    wt = np.r_[np.ones(len(yd)), np.full(ns, cfg["sc_total_weight"] * len(yd) / max(ns, 1))]
    team = lgb.LGBMClassifier(**params).fit(Xt, yt, sample_weight=wt)
    return alive, team


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "b1_ext.json"))
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config))

    # organisers' clarification (03/11): the simulator is calibrated on external tracking only
    # (src/poss/sim_external.py); the provided match is only predicted
    assert cfg.get("simulator") == "external", "only the externally calibrated simulator is supported"
    cal = cached(a.cache, "calibration", lambda: sim_external.variant(simulate.calibrate(
        sim_external.target([load("skillcorner", mid) for mid in cfg["sim_target_sc"]]),
        [load("dfl", mid) for mid in cfg["calibration_refs"]]), **cfg["sim_variant"]))
    if "domain_randomisation" in cfg:  # table measured on the clean external reference matches
        cfg["domain_randomisation"]["table"] = cached(a.cache, "dr_table", lambda: sim_external.dr_table(
            cal, [load("dfl", mid) for mid in cfg["calibration_refs"]]))
    alive, team = fit(cfg, cal, a.cache)

    bar = barath.load()
    F, _, _, period, near = dataset.rows(bar)
    # probabilities rounded to 1e-4 before the decoder (no development score changes,
    # experiments/rounding_submitted.md): the decoded labels do not depend on summation order
    pa = np.round(alive.predict_proba(F)[:, 1], ROUND)
    ph = np.round(team.predict_proba(F)[:, 1], ROUND)
    codes10 = decode.decode(pa, ph, period, tuple(cfg["costs"]))
    codes = decode.to_native(codes10, near, bar.n_frames)
    n = GAMES[GAME]["n_video_frames"]
    assert len(codes) == n
    codes, fixed = published.apply(codes, GAME)  # FAQ.md: frames 110808-113741 are D
    print("published structure:", fixed)
    path = os.path.join(a.out, f"{a.team}.csv")
    pd.DataFrame({"frame": np.arange(n), "label": LABELS[codes]}).to_csv(
        path, index=False, lineterminator="\n")
    print(path, hashlib.sha256(open(path, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
