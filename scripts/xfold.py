"""Decoder selection excluding the evaluated fold.

For each development fold, the team-switch cost is chosen by the zone score (b1_compare) on the
saved held-out predictions of the OTHER folds only, and the fold's own held-out sets are scored
at that cost. This removes the direct bias of choosing and scoring the cost on the same sets. It
is not a strictly nested validation: the models that predicted the other folds were trained with
the evaluated fold's matches, a residual, indirect leak that is not quantified.

Also: `with_control(tag)` writes pred_<tag>_ctl.pkl, the same predictions with the DFL sets'
ground truth in the control reading (data/cache/b1/control/<mid>.npz).
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_compare  # noqa: E402
import b1_dev  # noqa: E402
from poss import dataset, evaluate  # noqa: E402

C_GRID = b1_compare.ZONE_C_TEAM
C_STOP = b1_compare.ZONE_C_STOP


def fold_of(key):
    for i, (dfl, sc) in enumerate(b1_dev.folds()):
        if key.endswith("_" + dfl) and key.startswith("dfl"):
            return i
        if key.startswith("sc_") and key[3:] in sc:
            return i
    raise KeyError(key)


def grid(tag):
    """Per held-out set and cost: spellF1 (DFL full, SkillCorner live)."""
    return {c: b1_compare.per_match(tag, (c, C_STOP, C_STOP)) for c in C_GRID}


def zone_pick(g, keys):
    rows = []
    for c in C_GRID:
        d = np.mean([g[c][k] for k in keys if k.startswith("dfl_sim")])
        s = np.mean([g[c][k] for k in keys if k.startswith("sc_")])
        rows.append((c, d, s))
    bd, bs = max(r[1] for r in rows), max(r[2] for r in rows)
    gap = [max(bd - d, bs - s) for _, d, s in rows]
    return rows[int(np.argmin(gap))][0]


def scores(tag, g=None):
    """Held-out set -> spellF1 at the cost chosen without its fold; and the cost per fold."""
    g = g or grid(tag)
    keys = list(g[C_GRID[0]])
    out, chosen = {}, {}
    for i in range(b1_dev.N_FOLDS):
        own = [k for k in keys if fold_of(k) == i]
        if not own:
            continue
        c = zone_pick(g, [k for k in keys if fold_of(k) != i])
        chosen[i] = c
        for k in own:
            out[k] = g[c][k]
    return out, chosen


def summary(s):
    d = np.mean([v for k, v in s.items() if k.startswith("dfl_sim")])
    sc = np.mean([v for k, v in s.items() if k.startswith("sc_")])
    return d, sc, evaluate.combined(d, sc)


def paired(a, b, prefix):
    keys = [k for k in a if k.startswith(prefix)]
    d = np.array([a[k] - b[k] for k in keys])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d


def combined_paired(a, b):
    """Difference of the combined score and its paired SE (DFL and SC sets are independent)."""
    w = evaluate.SC_WEIGHT
    md, sd, _ = paired(a, b, "dfl_sim")
    ms, ss, _ = paired(a, b, "sc_")
    return (md + w * ms) / (1 + w), np.sqrt(sd ** 2 + (w * ss) ** 2) / (1 + w)


def with_control(tag):
    out = tag + "_ctl"
    path = os.path.join(b1_dev.CACHE, f"pred_{out}.pkl")
    if os.path.exists(path):
        return out
    pred = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))
    cache = {}
    for k, r in pred.items():
        if k.startswith("dfl"):
            mid = k.split("_")[-1]
            if mid not in cache:
                m = b1_dev.load("dfl", mid)
                team = np.load(os.path.join(b1_dev.CACHE, "control", f"{mid}.npz"),
                               allow_pickle=True)["team"]
                cache[mid] = dataset.codes(m.alive, team, m.period)
            assert len(r["gt"]) == len(cache[mid])
            r["gt"] = cache[mid]
    pickle.dump(pred, open(path, "wb"))
    return out
