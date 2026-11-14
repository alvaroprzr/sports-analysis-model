"""E7 against each of its members on the out-of-fold predictions of the final models (external
simulator), with the submitted decoder (configs/b2_ens_dr.json). Cached predictions of
scripts/f4_infer.py; external data only.

    POSS_SIM=ext_dr python scripts/b2_members_final.py > experiments/b2_e7_final.md
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
import xfold  # noqa: E402
from poss import ballout, decode, evaluate  # noqa: E402

CONF = json.load(open(os.path.join(ROOT, "configs", "b2_ens_dr.json")))
_BALL = {}


def ball(k):
    if k not in _BALL:
        _BALL[k] = (b2_dev.dfl_copy(k.split("_")[-1], b1_dev.TEST_SEED, False)["ball"]
                    if k.startswith("dfl") else b2_dev.sc_copy(k[3:], False)["ball"])
    return _BALL[k]


def scores(name):
    pred = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{name}.pkl"), "rb"))
    flag, ctl = {}, {}
    for k, r in pred.items():
        if k.startswith("dfl_clean"):
            continue
        pa = np.round(ballout.apply(r["p_alive"], ball(k), **CONF["ball_out"]), 4)
        nat = decode.to_native(decode.decode(pa, np.round(r["p_home"], 4), r["period"],
                                             tuple(CONF["costs"])), r["near"], r["n"])
        if k.startswith("dfl"):
            flag[k] = evaluate.full_scores(r["gt"], nat)["spellF1"]
            ctl[k] = evaluate.full_scores(b2_dev.control_codes(k.split("_")[-1])[0], nat)["spellF1"]
        else:
            flag[k] = ctl[k] = evaluate.live_scores(r["gt"], nat)["spellF1_live"]
    return flag, ctl


res = {n: scores(n) for n in ["f4d_swap"] + [f"f4d_swap_b2e{m}d" for m in "0125"]}
ens = res["f4d_swap"]
print("| model | DFL sim, flag | DFL sim, control | SC live | combined (mean of readings) | E7 minus it (paired SE) |")
print("|---|---|---|---|---|---|")
for n, (f, c) in res.items():
    df, sc = xfold.summary(f)[:2]
    dc = xfold.summary(c)[0]
    comb = (evaluate.combined(df, sc) + evaluate.combined(dc, sc)) / 2
    if n == "f4d_swap":
        diff = ""
    else:
        (d1, s1), (d2, s2) = xfold.combined_paired(ens[0], f), xfold.combined_paired(ens[1], c)
        diff = f"{(d1 + d2) / 2:+.4f} ({(s1 + s2) / 2:.4f})"
    print(f"| {n} | {df:.3f} | {dc:.3f} | {sc:.3f} | {comb:.4f} | {diff} |")
