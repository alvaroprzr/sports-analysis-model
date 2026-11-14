"""From a `Match` to model rows at the 10 Hz working rate.

Physics is computed at the native rate and aggregated (physics.to_rate); temporal context is
added as NaN-aware means over centred, past and future windows (Lang et al. 2022 found context up
to tens of seconds useful for ball-in-play detection).
"""

import numpy as np

from . import physics
from .kin import RATE
from .match import Match

# Params
CENTRED_S = (1, 5, 20, 60)  # centred mean windows
SIDE_S = (2, 8)  # past-only and future-only mean windows

H, A, D, X = 0, 1, 2, 3  # same codes as evaluation/possession_metrics.py


def swap_teams(m):
    """The same match with home and away exchanged, labels included (data augmentation)."""
    team = np.where(m.team >= 0, 1 - m.team, m.team).astype(np.int8)
    meta = dict(m.meta)
    if "detected_home" in meta:
        meta["detected_home"], meta["detected_away"] = meta["detected_away"], meta["detected_home"]
    return Match(m.name, m.source + "+swap", m.fps, m.pitch, m.away, m.home, m.ball, m.period,
                 m.alive, team, meta)


def codes(alive, team, period):
    """Ground-truth codes: H / A while alive with a known team, D when dead, X otherwise."""
    c = np.full(len(alive), X, np.int8)
    c[(alive == 1) & (team == 0)] = H
    c[(alive == 1) & (team == 1)] = A
    c[alive == 0] = D
    c[period == 0] = X
    return c


def _window_means(Z, period, lo, hi):
    """Mean of Z over frames [t + lo, t + hi] inside the same period, ignoring NaN."""
    out = np.full_like(Z, np.nan)
    for per in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == per)
        z = Z[idx]
        ok = np.isfinite(z)
        cs = np.vstack([np.zeros((1, z.shape[1])), np.cumsum(np.where(ok, z, 0), axis=0)])
        cn = np.vstack([np.zeros((1, z.shape[1])), np.cumsum(ok, axis=0)])
        n = len(idx)
        a = np.clip(np.arange(n) + lo, 0, n)
        b = np.clip(np.arange(n) + hi + 1, 0, n)
        s = cs[b] - cs[a]
        c = cn[b] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(c > 0, s / c, np.nan)
    return out


def rows(m, rate=RATE, n_rows=None):
    """Feature matrix at `rate` Hz (10 by default), its column names, labels, period and the
    native frame of each row. n_rows: keep only the first n_rows rows before the context windows
    (at most one fewer; used for SkillCorner, whose development rows end one native frame early,
    as the token rows of src/poss/tokens.py do)."""
    Xn, names = physics.frame_features(m)
    X10, near = physics.to_rate(Xn, names, m.fps, rate=rate)
    if n_rows is not None:
        assert 0 <= len(near) - n_rows <= 1, (len(near), n_rows)
        X10, near = X10[:n_rows], near[:n_rows]
    period = m.period[near]
    blocks, cols = [X10], list(names)
    for w in CENTRED_S:
        h = int(round(w * rate / 2))
        blocks.append(_window_means(X10, period, -h, h))
        cols += [f"{n} | c{w}s" for n in names]
    for w in SIDE_S:
        k = int(round(w * rate))
        blocks.append(_window_means(X10, period, -k, -1))
        cols += [f"{n} | past{w}s" for n in names]
        blocks.append(_window_means(X10, period, 1, k))
        cols += [f"{n} | next{w}s" for n in names]
    F = np.hstack(blocks).astype(np.float32)
    y = codes(m.alive[near], m.team[near], period)
    return F, cols, y, period, near
