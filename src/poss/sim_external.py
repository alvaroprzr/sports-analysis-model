"""Tracking simulator calibrated on external data only (organisers' clarification of 03/11: training
and tuning only on external data; no fitting or calibrating on the provided match).

The calibration code of simulate.py is applied to SkillCorner matches (real broadcast tracking with
its own defects: the ball missing, players leaving the camera view) instead of the provided match:
each match is upsampled to 25 fps (the DFL rate the simulator works at), players SkillCorner did not
detect are set missing (its extrapolated positions are not observations), and the matches are
concatenated with a stoppage between them. Intensity variants of the calibrated simulator are then
compared by the downstream score on SkillCorner (scripts/sim_ext.py), never on the provided match.
"""

import copy

import numpy as np

from . import kin, simulate
from .match import Match

FPS = 25.0
SEP_S = 4  # frames of no play between concatenated matches


def target(matches):
    """One 25 fps Match made of SkillCorner matches with undetected players missing."""
    parts = []
    for m in matches:
        home, away = m.home.copy(), m.away.copy()
        home[~m.meta["detected_home"]] = np.nan
        away[~m.meta["detected_away"]] = np.nan
        mm = Match(m.name, m.source, m.fps, m.pitch, home, away, m.ball, m.period, m.alive, m.team)
        parts.append(kin.resample(mm, FPS))
    n_home = max(p.home.shape[1] for p in parts)
    n_away = max(p.away.shape[1] for p in parts)
    pad = lambda x, n: np.concatenate([x, np.full((len(x), n - x.shape[1], 2), np.nan, np.float32)], 1)
    sep = int(SEP_S * FPS)
    cat = lambda xs, fill: np.concatenate(sum([[x, np.full((sep,) + x.shape[1:], fill, x.dtype)]
                                               for x in xs], []))
    home = cat([pad(p.home, n_home) for p in parts], np.nan)
    away = cat([pad(p.away, n_away) for p in parts], np.nan)
    ball = cat([p.ball for p in parts], np.nan)
    period = cat([p.period for p in parts], 0)
    alive = cat([p.alive for p in parts], -1)
    team = cat([p.team for p in parts], -1)
    return Match("skillcorner_target", "skillcorner", FPS, parts[0].pitch, home, away, ball,
                 period.astype(np.int8), alive.astype(np.int8), team.astype(np.int8))


def variant(cal, gaps=1.0, tracks=1.0, smooth=1.0):
    """Intensity variant: ball-gap onset, track-death hazard and smoothing widths multiplied."""
    c = copy.copy(cal)
    c.onset_scale = cal.onset_scale * gaps
    c.death_scale = cal.death_scale * tracks
    c.ball_sigma = cal.ball_sigma * smooth
    c.player_sigma = cal.player_sigma * smooth
    return c


# Domain randomisation (the arm chosen on SkillCorner, experiments/sim_ext_select2.md): each DFL
# training copy gets its own intensity. The share of visible ball is drawn uniformly in DR_VIS and
# turned into a ball-gap onset multiplier through a table measured on clean external DFL matches;
# the track-cut hazard is multiplied by U(DR_TRACKS). No figure of the provided match is used.
DR_MULTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
DR_VIS = (0.2, 1.0)
DR_TRACKS = (0.0, 3.0)


def dr_table(cal, refs, mults=DR_MULTS):
    """Gap-onset multiplier -> share of visible ball in play frames, on the reference matches."""
    share = [float(np.mean([np.isfinite(simulate.degrade(m, variant(cal, gaps=g),
                                                         np.random.default_rng(0))
                                        .ball[m.period > 0, 0]).mean() for m in refs]))
             for g in mults]
    return list(mults), share


def dr_copy(cal, table, mid, seed, vis=DR_VIS, tracks=DR_TRACKS):
    """Simulator of one training copy of DFL match `mid` (seeded by the copy, not by data)."""
    r = np.random.default_rng([seed, int(mid[3:], 36) % (2 ** 31)])
    v = r.uniform(*vis)
    mult, share = table  # visible share decreasing with the multiplier
    gaps = float(np.interp(-v, -np.asarray(share), np.asarray(mult)))
    return variant(cal, gaps=gaps, tracks=r.uniform(*tracks))
