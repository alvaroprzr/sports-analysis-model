"""F5: ball-out evidence (src/poss/ballout.py) on top of the submitted decoder (team-swapped E7,
minimum possession duration 5 rows), on the development held-out sets.

The triple (team-switch cost, min_out_m, min_s) is chosen by the zone score over three sources
(simulated DFL with the flag, with the control reading, SkillCorner live), excluding the evaluated
fold; the reference chooses its cost the same way without the evidence. Adoption: the F4 rule
(combined score averaged over the two readings better by more than 2 paired SE, no source worse by
more than 1 paired SE in either reading). Also the choice on all folds, for the submission.

    python scripts/f5_ballout.py > experiments/f5_ballout.md
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
import xfold  # noqa: E402
from poss import ballout, decode, decode2, evaluate  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "f4i_swap"
K = 5
C_GRID = (2, 3, 4, 5, 6, 8)
M_GRID = (0.5, 1.0, 2.0, 3.0)
S_GRID = (0.3, 0.5, 1.0, 2.0)
W = evaluate.SC_WEIGHT


def ball_tokens(keys):
    out = {}
    for k in keys:
        mid = k.split("_")[-1]
        if k.startswith("dfl_sim"):
            c = b2_dev.dfl_copy(mid, b1_dev.TEST_SEED, False)
        elif k.startswith("dfl_clean"):
            c = b2_dev.dfl_copy(mid, 0, False, clean=True)
        else:
            c = b2_dev.sc_copy(mid, False)
        out[k] = c["ball"]
    return out


def grid(pred, balls, params):
    g = {}
    for c in C_GRID:
        for q in params:
            s = {}
            for k, r in pred.items():
                pa = r["p_alive"] if q is None else ballout.apply(r["p_alive"], balls[k], *q)
                codes = decode2.decode(pa, r["p_home"], r["period"], (c, 6.0, 6.0), k=K)
                codes = decode.to_native(codes, r["near"], r["n"])
                s[k] = (evaluate.full_scores(r["gt"], codes)["spellF1"] if k.startswith("dfl")
                        else evaluate.live_scores(r["gt"], codes)["spellF1_live"])
            g[(c, q)] = s
    return g


def zone3(gf, gc, keys):
    rows = []
    for cq in gf:
        rows.append((cq, np.mean([gf[cq][k] for k in keys if k.startswith("dfl_sim")]),
                     np.mean([gc[cq][k] for k in keys if k.startswith("dfl_sim")]),
                     np.mean([gf[cq][k] for k in keys if k.startswith("sc_")])))
    best = [max(r[i] for r in rows) for i in (1, 2, 3)]
    gaps = [max(best[0] - a, best[1] - b, best[2] - c) for _, a, b, c in rows]
    return rows[int(np.argmin(gaps))][0]


def xscores(gf, gc):
    keys = list(next(iter(gf.values())))
    out = {"flag": {}, "control": {}}
    chosen = {}
    for i in range(b1_dev.N_FOLDS):
        cq = zone3(gf, gc, [k for k in keys if xfold.fold_of(k) != i])
        chosen[i] = cq
        for k in keys:
            if xfold.fold_of(k) == i:
                out["flag"][k], out["control"][k] = gf[cq][k], gc[cq][k]
    return out, chosen


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def main():
    pf = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{TAG}.pkl"), "rb"))
    pc = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{xfold.with_control(TAG)}.pkl"), "rb"))
    balls = ball_tokens(list(pf))
    params = [(m, s) for m in M_GRID for s in S_GRID]
    ref_f, ref_c = grid(pf, balls, [None]), grid(pc, balls, [None])
    new_f, new_c = grid(pf, balls, params), grid(pc, balls, params)
    ref, ch_r = xscores(ref_f, ref_c)
    new, ch_n = xscores(new_f, new_c)
    print(f"Reference ({TAG}, min duration {K}): costs per fold {[c for c, _ in ch_r.values()]}")
    print(f"Ball-out evidence: (cost, (min_out_m, min_s)) per fold {list(ch_n.values())}\n")
    print("| decoder | reading | DFL sim | SC live | combined |\n|---|---|---|---|---|")
    for name, s in (("reference", ref), ("ball-out", new)):
        for reading in ("flag", "control"):
            d, sc, cmb = xfold.summary(s[reading])
            print(f"| {name} | {reading} | {d:.3f} | {sc:.3f} | {cmb:.3f} |")
    keys = list(ref["flag"])
    diff = {r: {k: new[r][k] - ref[r][k] for k in keys} for r in ("flag", "control")}
    dd = np.array([(diff["flag"][k] + diff["control"][k]) / 2 for k in keys if k.startswith("dfl_sim")])
    ds = np.array([(diff["flag"][k] + diff["control"][k]) / 2 for k in keys if k.startswith("sc_")])
    cmb = (dd.mean() + W * ds.mean()) / (1 + W)
    s_c = np.sqrt(se(dd) ** 2 + (W * se(ds)) ** 2) / (1 + W)
    ok = cmb > 2 * s_c
    print(f"\nCombined, mean of the two readings: {cmb:+.4f} (paired SE {s_c:.4f})")
    for reading in ("flag", "control"):
        for prefix, label in (("dfl_sim", "DFL sim"), ("sc_", "SC live")):
            x = np.array([v for k, v in diff[reading].items() if k.startswith(prefix)])
            ok &= x.mean() >= -se(x)
            print(f"- {label}, {reading}: {x.mean():+.4f} (SE {se(x):.4f})")
    print(f"\nAdopt: {'YES' if ok else 'no'}")
    allk = list(new_f[next(iter(new_f))])
    print(f"Choice on all development folds: {zone3(new_f, new_c, allk)} "
          f"(reference cost {zone3(ref_f, ref_c, allk)[0]})")


if __name__ == "__main__":
    main()
