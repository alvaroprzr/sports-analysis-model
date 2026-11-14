"""Does E7 give the possession after a restart to the right team? Simulated DFL, control reading,
development held-out sets, E7 at the cost chosen excluding the evaluated fold (plain decoder):
team accuracy on the frames of the first 3 s after each true restart (end of a true stoppage),
by restart type (the DFL restart event within 5 s), against the team accuracy on all live frames.
Decides whether rules on who restarts can help (they can only fix the team, not the stoppage).

    python scripts/restart_team_check.py > experiments/restart_team_check.md
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import dead_ball_check as dbc  # noqa: E402
import xfold  # noqa: E402
from poss import dataset, decode  # noqa: E402

WINDOW = 3 * 25


def main():
    tag = xfold.with_control("ens_e7")
    _, chosen = xfold.scores(tag)
    pred = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))
    live_ok = live_n = 0
    by_type = {}
    for k, r in pred.items():
        if not k.startswith("dfl_sim"):
            continue
        c = chosen[xfold.fold_of(k)]
        p = decode.to_native(decode.decode(r["p_alive"], r["p_home"], r["period"],
                                           (c, xfold.C_STOP, xfold.C_STOP)), r["near"], r["n"])
        g = r["gt"]
        live = (g == dataset.H) | (g == dataset.A)
        both = live & ((p == dataset.H) | (p == dataset.A))
        live_ok += int((p[both] == g[both]).sum())
        live_n += int(both.sum())
        rs = dbc.restart_frames(k.split("_")[-1])
        rf = np.array([f for f, _ in rs])
        for a, b in dbc.runs((g == dataset.D)):
            if b >= len(g) or not live[b]:
                continue
            seg = slice(b, min(b + WINDOW, len(g)))
            m = live[seg] & ((p[seg] == dataset.H) | (p[seg] == dataset.A))
            if not m.any():
                continue
            near = np.flatnonzero(np.abs(rf - b) <= 5 * 25) if len(rf) else np.array([], int)
            typ = rs[near[np.argmin(np.abs(rf[near] - b))]][1] if len(near) else "other"
            acc = float((p[seg][m] == g[seg][m]).mean())
            by_type.setdefault(typ, []).append(acc)
    print(f"Team accuracy on all live frames (both live): {live_ok / live_n:.3f}\n")
    print("| restart type | restarts | team accuracy, first 3 s | restarts with accuracy < 0.5 |")
    print("|---|---|---|---|")
    allv = []
    for typ, v in sorted(by_type.items()):
        allv += v
        print(f"| {typ} | {len(v)} | {np.mean(v):.3f} | {int((np.array(v) < 0.5).sum())} |")
    print(f"| all | {len(allv)} | {np.mean(allv):.3f} | {int((np.array(allv) < 0.5).sum())} |")


if __name__ == "__main__":
    main()
