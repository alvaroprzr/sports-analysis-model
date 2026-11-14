"""How the sources differ beyond the headline statistics (identity stability, coverage, ball noise)
and what working at a lower frame rate costs on the boundaries of the labels themselves.

All quantities are aggregates; the provided match contributes label-free ones only.

    python scripts/dataset_audit.py > experiments/dataset_audit.md
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from poss import barath  # noqa: E402
from poss.match import Match  # noqa: E402

# Params
# Top human sprint speed is about 12.4 m/s (100 m world record splits); a frame-to-frame jump
# faster than this is a tracking artefact (identity swap or re-acquisition), not running.
V_MAX = 12.4
# A ball that moves with its nearest player within this distance is taken as "probably at the
# feet" for the label-free ball-noise comparison (PathCRF/Ball Radar use 1-2 m possession radii).
NEAR_M = 1.0
RATES = (12.5, 10.0, 5.0)


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def identity(m):
    P = np.concatenate([m.home, m.away], axis=1)
    play = m.period > 0
    step = np.linalg.norm(np.diff(P, axis=0), axis=2)
    ok = play[1:] & play[:-1]
    jumps = (step > V_MAX / m.fps)[ok].sum()
    player_min = np.isfinite(P[play, :, 0]).sum() / m.fps / 60
    frag = np.concatenate([runs(np.isfinite(P[play, i, 0])) for i in range(P.shape[1])]) / m.fps
    frag = frag[frag > 0]
    return {
        "jumps > 12.4 m/s per player-hour": 60 * jumps / player_min,
        "track fragments per slot-hour": len(frag) / (play.sum() / m.fps / 3600) / P.shape[1],
        "track fragment median s": np.median(frag),
        "slots in metadata": P.shape[1],
    }


def ball_noise(m):
    """Ball speed and ball-minus-player speed while the ball sits within NEAR_M of a player."""
    P = np.concatenate([m.home, m.away], axis=1)
    B = m.ball[:, :2]
    play = (m.period[1:] > 0) & (m.period[:-1] > 0)
    d = np.linalg.norm(P - B[:, None], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    j = np.argmin(d, axis=1)
    near = (d[np.arange(len(d)), j] < NEAR_M)[1:] & play
    vb = np.diff(B, axis=0) * m.fps
    k = np.arange(len(P) - 1)
    vp = (P[1:][k, j[1:]] - P[:-1][k, j[1:]]) * m.fps  # velocity of the player nearest at t+1
    vrel = np.linalg.norm(vb - vp, axis=1)
    ok = near & np.isfinite(vrel)
    return {
        "ball near a player (share of visible)": near[np.isfinite(vb[:, 0])].mean(),
        "ball speed | near, median m/s": np.median(np.linalg.norm(vb[ok], axis=1)),
        "ball-player rel. speed | near, median m/s": np.median(vrel[ok]),
    }


def codes(m):
    c = np.where(m.alive == 1, m.team, pm.D).astype(np.int8)
    c[(m.period == 0) | (m.alive < 0)] = pm.X
    c[(m.alive == 1) & (m.team < 0)] = pm.X
    return c


def resample_cost(m):
    """spellF1 of the labels themselves after sampling at a lower rate and holding back to 25 fps."""
    gt = codes(m)
    rn = pm.valid_runs(gt)
    out = {}
    for r in RATES:
        step = m.fps / r
        src = np.minimum(np.round(np.round(np.arange(m.n_frames) / step) * step).astype(int),
                         m.n_frames - 1)
        out[f"labels at {r:g} Hz -> spellF1"] = pm.spell_scores(gt, gt[src], rn)["mean_f1"]
    return out


def mean_rows(dicts):
    return {k: float(np.nanmean([d[k] for d in dicts])) for k in dicts[0]}


def main():
    split = json.load(open(os.path.join(ROOT, "configs", "splits.json")))
    bar = barath.load()
    rows = {"BAR-ATH": {**identity(bar), **ball_noise(bar)}}
    for src, key in (("dfl_v3", "dfl"), ("skillcorner_v3", "skillcorner")):
        ms = [Match.load(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "external",
                                                                   src, "*.npz")))
              if os.path.basename(f)[:-4] in split[key]["dev"]]
        per = []
        for m in ms:
            r = {**identity(m), **ball_noise(m)}
            if src == "dfl_v3":
                r.update(resample_cost(m))
            if "detected_home" in m.meta:
                det = np.concatenate([m.meta["detected_home"], m.meta["detected_away"]], axis=1)
                fin = np.isfinite(np.concatenate([m.home, m.away], axis=1)[..., 0])
                r["positions actually detected"] = det[fin].mean()
            per.append(r)
        rows[f"{key} dev (n={len(ms)})"] = mean_rows(per)
    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    print("| statistic | " + " | ".join(rows) + " |")
    print("|---" * (len(rows) + 1) + "|")
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{r.get(k, np.nan):.3g}" for r in rows.values()) + " |")


if __name__ == "__main__":
    main()
