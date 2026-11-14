"""Derive the two physical constants of the features from external data only (DFL and
SkillCorner), to check them against the values in src/poss/physics.py.

- V_REACH: 99th percentile of player speed at 10 Hz in open play (same computation as
  scripts/domain_stats.py), per source as the mean of the per-match percentiles, and pooled over
  every frame of both sources; rounded to the whole m/s.
- IMPULSE_S: the ball smoothing width of broadcast tracking, from the external calibration on
  SkillCorner (experiments/sim_ext_calibration.md, ball_sigma_full, in frames at 25 fps): one and
  two widths.

    python scripts/derive_constants.py > experiments/derive_constants.md
"""

import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from poss import physics  # noqa: E402
from poss.match import Match  # noqa: E402

# Params
RATE = 10.0
SOURCES = ("dfl_v3", "skillcorner_v3")
CALIB = os.path.join(ROOT, "experiments", "sim_ext_calibration.md")
CALIB_FPS = 25.0


def speeds(m):
    """Player speeds (m/s) at 10 Hz between consecutive open-play rows."""
    idx = np.round(np.arange(0, m.n_frames, m.fps / RATE)).astype(int)
    idx = idx[idx < m.n_frames]
    play = m.period[idx] > 0
    P = np.concatenate([m.home, m.away], axis=1)[idx]
    v = np.linalg.norm(np.diff(P, axis=0), axis=2) * RATE
    ok = play[1:] & play[:-1]
    vv = v[ok]
    return vv[np.isfinite(vv)]


def main():
    print("Physical constants derived from external data only (no file of the provided match is read).\n")
    print("| source | matches | p99 player speed (mean of matches) | pooled p99 | rounded |")
    print("|---|---|---|---|---|")
    pooled = []
    for src in SOURCES:
        files = sorted(glob.glob(os.path.join(ROOT, "data", "external", src, "*.npz")))
        per, allv = [], []
        for f in files:
            v = speeds(Match.load(f))
            per.append(np.percentile(v, 99))
            allv.append(v)
        allv = np.concatenate(allv)
        pooled.append(allv)
        p = np.percentile(allv, 99)
        print(f"| {src} | {len(files)} | {np.mean(per):.2f} | {p:.2f} | {np.mean(per):.0f} |")
    both = np.concatenate(pooled)
    p = np.percentile(both, 99)
    print(f"| both, pooled frames | - | - | {p:.2f} | {p:.0f} |")
    print(f"\nV_REACH in src/poss/physics.py: {physics.V_REACH}")

    txt = open(CALIB).read()
    sigma = float(re.search(r"ball_sigma_full:\s*([0-9.]+)", txt).group(1))
    s = sigma / CALIB_FPS
    print(f"\nBall smoothing width (external calibration on SkillCorner, {os.path.relpath(CALIB, ROOT)}):"
          f" {sigma:g} frames at 25 fps = {s:.2f} s; one and two widths: ({s:.1f}, {2 * s:.1f}) s")
    print(f"IMPULSE_S in src/poss/physics.py: {physics.IMPULSE_S}")


if __name__ == "__main__":
    main()
