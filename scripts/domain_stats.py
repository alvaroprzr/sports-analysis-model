"""Aggregate statistics of every source, to measure how far the external tracking is from ours.

Only label-free quantities are computed for the provided match. Everything is resampled to 10 Hz
(the lowest native rate among the sources) so speeds and jitter are comparable.

    python scripts/domain_stats.py > experiments/domain_stats.md
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from poss import barath  # noqa: E402
from poss.match import Match  # noqa: E402

# Params
RATE = 10.0


def runs(mask):
    """Lengths of the runs of True in a boolean array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def stats(m):
    idx = np.round(np.arange(0, m.n_frames, m.fps / RATE)).astype(int)
    idx = idx[idx < m.n_frames]
    play = m.period[idx] > 0
    P = np.concatenate([m.home, m.away], axis=1)[idx]
    B = m.ball[idx]
    out = {}
    miss = np.isnan(B[:, 0])
    out["ball missing"] = miss[play].mean()
    g = runs(miss & play) / RATE
    out["ball gap median s"] = np.median(g) if len(g) else np.nan
    out["ball gaps per min"] = len(g) / (play.sum() / RATE / 60)
    out["ball gaps <1s share"] = (g < 1).mean() if len(g) else np.nan
    out["players tracked"] = np.isfinite(P[play, :, 0]).sum(1).mean()
    v = np.linalg.norm(np.diff(P, axis=0), axis=2) * RATE
    ok = play[1:] & play[:-1]
    vv = v[ok][np.isfinite(v[ok])]
    out["player speed median"] = np.median(vv)
    out["player speed p99"] = np.percentile(vv, 99)
    a = np.linalg.norm(np.diff(P, 2, axis=0), axis=2) * RATE ** 2
    aa = a[ok[1:]][np.isfinite(a[ok[1:]])]
    out["player accel p95"] = np.percentile(aa, 95)
    vb = np.linalg.norm(np.diff(B[:, :2], axis=0), axis=1) * RATE
    vb = vb[ok & np.isfinite(vb)]
    out["ball speed p50"] = np.median(vb)
    out["ball speed p99"] = np.percentile(vb, 99)
    d = np.nanmin(np.linalg.norm(P - B[:, None, :2], axis=2), axis=1)
    d = d[play & ~miss & np.isfinite(d)]
    out["ball-nearest player median m"] = np.median(d)
    if (m.alive >= 0).any():
        al = m.alive[idx]
        tm = m.team[idx]
        lab = play & (al >= 0)
        out["alive share"] = (al[lab] == 1).mean()
        out["ball missing | alive"] = miss[lab & (al == 1)].mean()
        out["ball missing | dead"] = miss[lab & (al == 0)].mean()
        code = np.where(al == 1, tm, 2)[lab]
        poss = (np.diff(code) != 0) & (code[1:] < 2)
        out["possessions per 5 min"] = (poss.sum() + (code[0] < 2)) / (lab.sum() / RATE / 300)
        seg = np.split(code, np.flatnonzero(np.diff(code)) + 1)
        dur = np.array([len(s) for s in seg if s[0] < 2]) / RATE
        out["possession median s"] = np.median(dur)
        out["possessions <4s share"] = (dur < 4).mean()
    return out


def main():
    rows = {}
    rows["BAR-ATH (no labels)"] = stats(barath.load())
    for src in ("dfl_v3", "skillcorner_v3"):
        files = sorted(glob.glob(os.path.join(ROOT, "data", "external", src, "*.npz")))
        per = [stats(Match.load(f)) for f in files]
        if per:
            keys = per[0].keys()
            rows[f"{src} (n={len(per)})"] = {k: np.nanmean([p.get(k, np.nan) for p in per])
                                             for k in keys}
    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    print("| statistic | " + " | ".join(rows) + " |")
    print("|---" * (len(rows) + 1) + "|")
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{r.get(k, np.nan):.3g}" for r in rows.values()) + " |")


if __name__ == "__main__":
    main()
