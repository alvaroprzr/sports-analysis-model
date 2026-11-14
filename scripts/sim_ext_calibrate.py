"""Calibrate the external simulator (src/poss/sim_external.py) and save its intensity variants as
data/cache/sim_calibration_<name>.pkl, used by the development scripts when POSS_SIM=<name>.
No data of the provided match is read.

    python scripts/sim_ext_calibrate.py > experiments/sim_ext_calibration.md
"""

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
from poss import sim_external, simulate  # noqa: E402

# Params
TARGET_SC = 3  # SkillCorner development matches the calibration statistics come from (first ones)
REFS = ["J03WMX", "J03WN1"]  # clean DFL matches used to match those statistics (as before)
VARIANTS = {"ext": {}, "ext_g2": {"gaps": 2.0}, "ext_g05": {"gaps": 0.5},
            "ext_t2": {"tracks": 2.0}}


def main():
    sc = b1_dev.SPLIT["skillcorner"]["dev"][:TARGET_SC]
    tgt = sim_external.target([b1_dev.load("skillcorner", m) for m in sc])
    cal = simulate.calibrate(tgt, [b1_dev.load("dfl", m) for m in REFS])
    print(f"External simulator: calibration statistics from SkillCorner {', '.join(sc)} "
          f"(25 fps, undetected players missing); clean DFL references {', '.join(REFS)}.\n")
    for k, v in cal.stats.items():
        print(f"- {k}: {v}")
    print()
    for name, kw in VARIANTS.items():
        c = sim_external.variant(cal, **kw)
        pickle.dump(c, open(os.path.join(ROOT, "data", "cache", f"sim_calibration_{name}.pkl"), "wb"))
        print(f"- saved {name}: {kw or 'as calibrated'}")


if __name__ == "__main__":
    main()
