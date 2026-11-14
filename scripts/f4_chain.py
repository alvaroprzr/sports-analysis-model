"""F4, part 3: decoder chains (src/poss/decode2.py) on the E7 development predictions, with the
pair (team-switch cost, variant parameter) chosen excluding the evaluated fold by the zone score:

    mindur   minimum possession duration k rows (k / 10 s), k in K_GRID (k = 1: plain decoder)
    varcost  team-switch cost per row c / (1 + alpha * p / mean(p)), p = E5's "possession changes
             here" probability (full inference, same weights as f4i_full), alpha in ALPHA_GRID

Reference: f4i_full (the members re-predicted with the saved weights, plain decoder).

    python scripts/f4_chain.py > experiments/f4_chain.md
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import xfold  # noqa: E402
from poss import decode, decode2, evaluate  # noqa: E402

K_GRID = (1, 5, 10, 20)
ALPHA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
REF = "f4i_full"


def load(tag):
    return pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))


def score(k, r, codes):
    c = decode.to_native(codes, r["near"], r["n"])
    return (evaluate.full_scores(r["gt"], c)["spellF1"] if k.startswith("dfl")
            else evaluate.live_scores(r["gt"], c)["spellF1_live"])


def grid(pred, variant, pchange):
    g = {}
    params = K_GRID if variant == "mindur" else ALPHA_GRID
    for c in xfold.C_GRID:
        for q in params:
            s = {}
            for k, r in pred.items():
                costs = (c, xfold.C_STOP, xfold.C_STOP)
                if variant == "mindur":
                    codes = decode2.decode(r["p_alive"], r["p_home"], r["period"], costs, k=q)
                else:
                    p = pchange[k]
                    tc = c / (1 + q * p / max(p.mean(), 1e-6))
                    codes = decode2.decode(r["p_alive"], r["p_home"], r["period"], costs, team_cost=tc)
                s[k] = score(k, r, codes)
            g[(c, q)] = s
    return g


def pick(g, keys):
    rows = []
    for cq, s in g.items():
        rows.append((cq, np.mean([s[k] for k in keys if k.startswith("dfl_sim")]),
                     np.mean([s[k] for k in keys if k.startswith("sc_")])))
    bd, bs = max(r[1] for r in rows), max(r[2] for r in rows)
    return rows[int(np.argmin([max(bd - d, bs - s) for _, d, s in rows]))][0]


def xscores(g):
    keys = list(next(iter(g.values())))
    out, chosen = {}, {}
    for i in range(b1_dev.N_FOLDS):
        own = [k for k in keys if xfold.fold_of(k) == i]
        cq = pick(g, [k for k in keys if xfold.fold_of(k) != i])
        chosen[i] = cq
        out.update({k: g[cq][k] for k in own})
    return out, chosen


def main():
    pchange = {k: r["p_change"] for k, r in load("f4i_full_b2e5").items()}
    res = {}
    for reading, tag in (("flag", REF), ("control", xfold.with_control(REF))):
        pred = load(tag)
        res[("ref", reading)] = xfold.scores(tag)
        for variant in ("mindur", "varcost"):
            res[(variant, reading)] = xscores(grid(pred, variant, pchange))
    print("| variant | reading | (c_team, parameter) per fold | DFL sim | SC live | combined |")
    print("|---|---|---|---|---|---|")
    for (v, reading), (s, ch) in res.items():
        d, sc, cmb = xfold.summary(s)
        print(f"| {v} | {reading} | {list(ch.values())} | {d:.3f} | {sc:.3f} | {cmb:.3f} |")
    print("\nPaired differences against the plain decoder (combined: diff (SE); DFL sim; SC live):\n")
    for v in ("mindur", "varcost"):
        cells = []
        for reading in ("flag", "control"):
            a, b = res[(v, reading)][0], res[("ref", reading)][0]
            diff, se = xfold.combined_paired(a, b)
            md, sd, _ = xfold.paired(a, b, "dfl_sim")
            ms, ss, _ = xfold.paired(a, b, "sc_")
            cells.append(f"{reading}: {diff:+.4f} ({se:.4f}), DFL {md:+.4f} ({sd:.4f}), "
                         f"SC {ms:+.4f} ({ss:.4f})")
        print(f"- {v}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
