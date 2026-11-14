"""Label-free sanity checks (report only):

1. Attacking direction per team and period, two ways: the centroid rule used by every feature
   (physics.attack_sign) and the side of each team's deepest player (its goalkeeper, as a median
   over the period). Both must agree in every period of BAR-ATH, the DFL and SkillCorner.
2. What the submitted file predicts in the first and last 20 s of each graded half of BAR-ATH
   (label counts, changes, segments). Predictions only; no ground truth is read.

    python scripts/sanity_checks.py data/cache/b2ens2/a/b2ens.csv > experiments/sanity_checks.md
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import b1_dev  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from poss import barath, physics  # noqa: E402

EDGE_S = 20


def directions(m):
    """Per period: (centroid sign, deepest-player sign); +1 = home attacks +x, 0 = undecided."""
    out = {}
    sign = physics.attack_sign(m)
    for per in np.unique(m.period[m.period > 0]):
        f = m.period == per
        hx, ax = m.home[f, :, 0], m.away[f, :, 0]
        with np.errstate(all="ignore"):
            h_lo, h_hi = np.nanmedian(np.nanmin(hx, 1)), np.nanmedian(np.nanmax(hx, 1))
            a_lo, a_hi = np.nanmedian(np.nanmin(ax, 1)), np.nanmedian(np.nanmax(ax, 1))
        if -h_lo > h_hi and a_hi > -a_lo:
            gk = 1  # home's deepest player near -x, away's near +x
        elif h_hi > -h_lo and -a_lo > a_hi:
            gk = -1
        else:
            gk = 0
        out[int(per)] = (int(sign[f][0]), gk)
    return out


def main():
    print("## Attacking direction (+1: home attacks +x; 0: deepest-player rule undecided)\n")
    print("| source | match | period: centroid / deepest player | agree |\n|---|---|---|---|")
    cases = [("BAR-ATH", "BAR-ATH", barath.load)]
    cases += [("DFL", mid, lambda mid=mid: b1_dev.load("dfl", mid))
              for mid in b1_dev.SPLIT["dfl"]["dev"]]
    cases += [("SkillCorner", mid, lambda mid=mid: b1_dev.load("skillcorner", mid))
              for mid in b1_dev.SPLIT["skillcorner"]["dev"]]
    bad = 0
    for src, name, load in cases:
        d = directions(load())
        ok = all(c == g for c, g in d.values())
        bad += not ok
        print(f"| {src} | {name} | " + ", ".join(f"{p}: {c:+d} / {g:+d}" for p, (c, g) in d.items())
              + f" | {'yes' if ok else 'NO'} |", flush=True)
    print(f"\nMatches with a disagreeing period: {bad}.\n")

    if len(sys.argv) > 1:
        lab = pd.read_csv(sys.argv[1])["label"].to_numpy()
        n = EDGE_S * 25
        print(f"## Submitted file, first and last {EDGE_S} s of each graded half (predictions)\n")
        print("| half | edge | H | A | D | X | changes |\n|---|---|---|---|---|---|---|")
        for half, first, last in half_video_blocks(GAME):
            for edge, seg in (("first", lab[first:first + n]), ("last", lab[last + 1 - n:last + 1])):
                cnt = {k: int((seg == k).sum()) for k in "HADX"}
                print(f"| {half} | {edge} | {cnt['H']} | {cnt['A']} | {cnt['D']} | {cnt['X']} | "
                      f"{int((seg[1:] != seg[:-1]).sum())} |")
        print("\nSimulator: the ball and player smoothing is scipy's gaussian_filter1d (symmetric, "
              "acausal), applied in degrade() step 'smooth', before the ball gaps (step 'ball').")


if __name__ == "__main__":
    main()
