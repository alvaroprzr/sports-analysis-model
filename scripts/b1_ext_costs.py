"""Team-switch cost of the new B1 reserve: zone score over three sources (simulated DFL with the
flag, with the control reading, SkillCorner live), all development folds, plain decoder; and the
estimate with the cost chosen excluding the evaluated fold.

    POSS_SIM=<name> python scripts/b1_ext_costs.py TAG > experiments/b1_ext_costs.md
"""

import sys

import numpy as np

import b1_compare
import b1_dev
import xfold


def zone3(gf, gc, keys):
    rows = []
    for c in gf:
        rows.append((c, np.mean([gf[c][k] for k in keys if k.startswith("dfl_sim")]),
                     np.mean([gc[c][k] for k in keys if k.startswith("dfl_sim")]),
                     np.mean([gf[c][k] for k in keys if k.startswith("sc_")])))
    best = [max(r[i] for r in rows) for i in (1, 2, 3)]
    return rows, [max(best[0] - a, best[1] - b, best[2] - s) for _, a, b, s in rows]


def main():
    tag = sys.argv[1]
    gf, gc = xfold.grid(tag), xfold.grid(xfold.with_control(tag))
    keys = list(gf[b1_compare.ZONE_C_TEAM[0]])
    rows, gaps = zone3(gf, gc, keys)
    print("| c_team | DFL flag | DFL control | SC live | gap |\n|---|---|---|---|---|")
    for (c, a, b, s), g in zip(rows, gaps):
        print(f"| {c} | {a:.3f} | {b:.3f} | {s:.3f} | {g:.3f} |")
    print(f"\nChosen for the reserve: c_team {rows[int(np.argmin(gaps))][0]}")
    out = {"flag": {}, "control": {}}
    for i in range(b1_dev.N_FOLDS):
        r, g = zone3(gf, gc, [k for k in keys if xfold.fold_of(k) != i])
        c = r[int(np.argmin(g))][0]
        for k in keys:
            if xfold.fold_of(k) == i:
                out["flag"][k], out["control"][k] = gf[c][k], gc[c][k]
    for reading in ("flag", "control"):
        d, s, cmb = xfold.summary(out[reading])
        print(f"- excluding the evaluated fold, {reading}: DFL sim {d:.3f}, SC live {s:.3f}, combined {cmb:.3f}")


if __name__ == "__main__":
    main()
