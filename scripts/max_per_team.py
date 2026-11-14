"""Number of player slots per team in the tokens (tokens.MAX_PER_TEAM), fixed on external data only:
the largest number of players of one team tracked at the same time in any frame of the external
development and training matches (clean DFL, DFL degraded by the chosen simulator, SkillCorner with
undetected players counted as missing), reported per source.

    POSS_SIM=ext_dr python scripts/max_per_team.py > experiments/max_per_team.md
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
from poss import simulate  # noqa: E402


def most(m, det=None):
    out = 0
    for side, x in (("home", m.home), ("away", m.away)):
        seen = np.isfinite(x[..., 0])
        if det is not None:
            seen &= m.meta[f"detected_{side}"]
        out = max(out, int(seen[m.period > 0].sum(1).max()))
    return out


def main():
    dfl = sorted(set(sum([[d] for d, _ in b1_dev.folds()], [])) | set(b1_dev.SPLIT["dfl"].get("train", [])))
    sc = list(b1_dev.SPLIT["skillcorner"]["dev"]) + list(b1_dev.SPLIT["skillcorner"].get("train", []))
    rows = {"DFL clean": [], "DFL simulated (training copies)": [], "SkillCorner (detected)": []}
    for mid in dfl:
        m = b1_dev.load("dfl", mid)
        rows["DFL clean"].append(most(m))
        for seed in range(2):
            d = simulate.degrade(m, b1_dev.copy_calibration(mid, seed), np.random.default_rng(seed),
                                 spread=b1_dev.SPREAD)
            rows["DFL simulated (training copies)"].append(most(d))
    for mid in sc:
        rows["SkillCorner (detected)"].append(most(b1_dev.load("skillcorner", mid), det=True))
    print("| source | matches/copies | most players of one team tracked at once |\n|---|---|---|")
    for k, v in rows.items():
        print(f"| {k} | {len(v)} | {max(v)} |")
    print(f"\nMAX_PER_TEAM = {max(max(v) for v in rows.values())}")


if __name__ == "__main__":
    main()
