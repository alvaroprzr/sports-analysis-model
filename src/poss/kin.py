"""Resampling to the working rate and kinematics derived from positions.

Every source goes through the same functions, so velocities and accelerations mean the same thing
everywhere (no provider speed channel is ever used).
"""

import numpy as np

from .match import Match

# Params
RATE = 10.0  # Hz; the lowest native rate among the sources (SkillCorner)


def _interp(a, t):
    """Linear interpolation of a along axis 0 at fractional indices t; NaN if a neighbour is NaN."""
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(a) - 1)
    w = (t - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    return (1 - w) * a[i0] + w * a[i1]


def resample(m, rate=RATE):
    """`Match` at `rate` Hz. Positions are interpolated at exact times (nearest-frame picking from
    25 fps alternates 0.08 / 0.12 s steps and shows up as velocity noise); labels take the
    nearest frame."""
    if m.fps == rate:
        return m
    t = np.arange(0, m.n_frames - 1, m.fps / rate)
    near = np.round(t).astype(int)
    meta = {k: v[near] for k, v in m.meta.items() if isinstance(v, np.ndarray) and len(v) == m.n_frames}
    meta["src_frame"] = near
    return Match(m.name, m.source, rate, m.pitch, _interp(m.home, t).astype(np.float32),
                 _interp(m.away, t).astype(np.float32), _interp(m.ball, t).astype(np.float32),
                 m.period[near], m.alive[near], m.team[near], meta)


def velocity(x, fps):
    """Central difference (one-sided at the ends of a track); NaN where no neighbour exists."""
    v = np.full_like(x, np.nan)
    v[1:-1] = (x[2:] - x[:-2]) * (fps / 2)
    fwd = (x[1:] - x[:-1]) * fps
    edge = np.isnan(v[1:-1])
    v[1:-1][edge] = np.where(np.isnan(fwd[1:][edge]), fwd[:-1][edge], fwd[1:][edge])
    return v
