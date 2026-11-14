"""Domain-randomisation arm of the external simulator (POSS_SIM=ext_dr): the calibrated external
simulator plus a table from the ball-gap onset multiplier to the share of visible ball it gives,
measured on clean external DFL matches, so that each training copy can draw a visible share
uniformly in b1_dev.DR_VIS (from clean to very degraded; no figure of the provided match is used).

    python scripts/sim_ext_dr.py > experiments/sim_ext_dr.md
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
from poss import sim_external, simulate  # noqa: E402

MULTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
REFS = ["J03WMX", "J03WN1"]


def main():
    cal = pickle.load(open(os.path.join(ROOT, "data", "cache", "sim_calibration_ext.pkl"), "rb"))
    refs = [b1_dev.load("dfl", m) for m in REFS]
    share = []
    print("| gap-onset multiplier | visible ball share (play frames) |\n|---|---|")
    for g in MULTS:
        c = sim_external.variant(cal, gaps=g)
        v = np.mean([np.isfinite(simulate.degrade(m, c, np.random.default_rng(0))
                                 .ball[m.period > 0, 0]).mean() for m in refs])
        share.append(float(v))
        print(f"| {g:g} | {v:.3f} |", flush=True)
    pickle.dump({"cal": cal, "table": (list(MULTS), share)},
                open(os.path.join(ROOT, "data", "cache", "sim_calibration_ext_dr.pkl"), "wb"))
    print(f"\nRange covered: {min(share):.2f}-{max(share):.2f}; draws in {b1_dev.DR_VIS}, "
          f"track-cut multiplier in {b1_dev.DR_TRACKS}.")


if __name__ == "__main__":
    main()
