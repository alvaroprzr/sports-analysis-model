"""Regenerate the B2 combination submission from the saved weights of its members (CPU,
deterministic): the equal-weight average of the members' probabilities, rounded to 1e-4, then the
decoder of the configuration (costs, team swap, ball-out rule, minimum duration, boundary shifts).
With "alive" in the configuration, P(alive) is the stoppage classifier of scripts/alive_dev.py
(experiments/alive_select2.md) instead of the members' average; P(home | alive) stays the members'.

    python scripts/make_submission_b2ens.py --team NAME [--config configs/b2_ens_dr.json] [--threads N]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import b2_dev  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from poss import published, b2, ballout, barath, dataset, decode, decode2, shift, tokens  # noqa: E402

LABELS = np.array(["H", "A", "D", "X"])
ROUND = 4


def load_model(path):
    ck = torch.load(path, weights_only=False)
    cfg = ck["cfg"]
    model = b2.B2(b2_dev.N_CTX, d=cfg["d"], heads=cfg["heads"], hidden=cfg["hidden"],
                  social=cfg["social"], k_near=cfg["k_near"], temporal=cfg["temporal"],
                  change_head=cfg["change_head"], ball_head=cfg.get("ball_head", False))
    model.load_state_dict(ck["state"])
    return model, ck["stats"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "b2_ens_dr.json"))
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    b2.set_threads(a.threads)
    conf = json.load(open(a.config))
    costs = tuple(conf["costs"])

    bar = barath.load()
    pl, bl, near, dref = tokens.tokens(bar)
    F, cols, _, period, near_rows = dataset.rows(bar)
    assert np.array_equal(near, near_rows)
    copy = dict(players=pl, ball=bl, dref=dref, ctx=F[:, :b2_dev.N_CTX], period=period,
                y=np.zeros(len(period), np.int8))
    swap = conf.get("team_swap", False)
    if swap:
        # the same match with home and away exchanged (F4, experiments/f4d_rule.md)
        sw = dataset.swap_teams(bar)
        pl2, bl2, near2, dref2 = tokens.tokens(sw)
        F2, cols2, _, period2, near_rows2 = dataset.rows(sw)
        assert np.array_equal(near2, near) and np.array_equal(period2, period)
        copy2 = dict(players=pl2, ball=bl2, dref=dref2, ctx=F2[:, :b2_dev.N_CTX], period=period,
                     y=np.zeros(len(period), np.int8))
    pas, phs = [], []
    for name in sorted(conf["members"]):
        model, stats = load_model(os.path.join(ROOT, conf["members"][name]["weights"]))
        pa, ph = b2_dev.predict(model, copy, stats)
        if swap:
            pa2, ph2 = b2_dev.predict(model, copy2, stats)
            pa, ph = (pa + pa2) / 2, (ph + 1 - ph2) / 2
        pas.append(pa)
        phs.append(ph)
        del model
    pa = np.mean(pas, 0)
    al = conf.get("alive")
    if al:
        # P(alive) of the stoppage classifier on the same rows, mean of the match and of the match
        # with the teams exchanged, as it was selected (scripts/alive_select2.py)
        import lightgbm as lgb
        from alive_dev import keep_cols
        assert swap and list(cols2) == list(cols)
        keep = keep_cols(cols)
        # read as text: the model loads the same whatever line endings the checkout gave the file
        gbm = lgb.Booster(model_str=open(os.path.join(ROOT, al["model"]), encoding="ascii").read())
        assert gbm.num_feature() == int(keep.sum())
        pa = (gbm.predict(F[:, keep], num_threads=a.threads)
              + gbm.predict(F2[:, keep], num_threads=a.threads)) / 2
        assert len(pa) == len(period) and np.isfinite(pa).all()
    bo = conf.get("ball_out")
    if bo:
        # F5: ball visible outside the lines for a while -> P(alive) ~ 0 before decoding
        n_out = int(ballout.out_rows(bl, bo["min_out_m"], bo["min_s"]).sum())
        pa = ballout.apply(pa, bl, bo["min_out_m"], bo["min_s"])
        print("ball-out evidence rows:", n_out)
    pa = np.round(pa, ROUND)
    ph = np.round(np.mean(phs, 0), ROUND)
    k = int(conf.get("min_possession_rows", 1))
    codes10 = (decode.decode(pa, ph, period, costs) if k == 1
               else decode2.decode(pa, ph, period, costs, k=k))
    if conf.get("boundary_shift_s"):
        # fixed shift per boundary type at the working rate (src/poss/shift.py)
        codes10 = shift.apply(codes10, period, conf["boundary_shift_s"])
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
