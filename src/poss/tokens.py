"""Per-frame sets of player and ball tokens for the B2 model, at the 10 Hz working rate.

Everything is computed at the native rate with the same quantities as physics.py and sampled at
the 10 Hz rows of physics.to_rate. Players enter as an unordered set per frame (the tracked
players of each team, nearest to the ball first, padded with a mask): no token carries an identity
across frames, because BAR-ATH's tracks fragment every half minute on median.

Player token (per team, MAX_PER_TEAM slots): x, y in the attacking frame (/half pitch), velocity
(/8 m/s), distance to the ball (/10 m, 3 when the ball is missing), and, at the 0.4 s impulse
scale, whether the player stands behind the ball w.r.t. its outgoing direction and their speed
along it, the ball-minus-player speed (/10 m/s), and a present flag.
Ball token: x, y (/half pitch), velocity (/10 m/s), visible flag, impulse at 0.2 and 0.4 s (/10),
distance beyond a line (/5 m), gap position and log gap length (0 when visible).
"""

import numpy as np

from . import physics

# Params
MAX_PER_TEAM = 11  # most players of one team tracked at once in the external data (experiments/max_per_team.md), the laws of the game
SCALE_POS = (52.5, 34.0)
N_PLAYER_FEATS = 10
N_BALL_FEATS = 10
# Normalised inputs are clipped to +-4 (32 m/s for players, 40 m/s for the ball): tracking glitches
# (substitutions, re-acquisitions) otherwise produce speeds of hundreds of m/s.
CLIP = 4.0


def _impulse(B, fps, s):
    k = max(2, int(round(s * fps)))
    vbef, vaft = physics._window_velocity(B, k)
    return vbef * fps, vaft * fps


def reference_point(m):
    """Where play is, per native frame: the ball if visible, else its last known position in the
    same period (BAR-ATH loses the ball on 41 % of frames, mostly in stoppages where it stays
    put), else the centroid of all tracked players (before the first sighting of a period)."""
    B = m.ball[:, :2].astype(np.float64)
    P = np.concatenate([m.home, m.away], axis=1).astype(np.float64)
    ref = B.copy()
    for per in np.unique(m.period):
        idx = np.flatnonzero(m.period == per)
        last = None
        for t in idx:
            if np.isfinite(B[t, 0]):
                last = B[t]
            elif last is not None:
                ref[t] = last
    miss = ~np.isfinite(ref[:, 0])
    with np.errstate(all="ignore"):
        ref[miss] = np.nanmean(P[miss], axis=1)
    return ref


def tokens(m):
    """(players (T10, 2, MAX, F) float16, ball (T10, F) float16, near, dref (T10, 2, MAX) float16)
    for match m. Players of each team are ordered by distance to `reference_point`, which is also
    returned (metres / 10, 99 for empty slots) so a model can keep the nearest ones."""
    P_home, P_away = m.home.astype(np.float64), m.away.astype(np.float64)
    B = m.ball[:, :2].astype(np.float64)
    fps = m.fps
    T = m.n_frames
    sgn = physics.attack_sign(m).astype(np.float64)
    t10 = np.arange(0, T - 1, fps / physics.RATE)
    near = np.round(t10).astype(int)
    n = len(near)

    vb = np.full_like(B, np.nan)
    vb[1:-1] = (B[2:] - B[:-2]) * fps / 2
    vbef4, vaft4 = _impulse(B, fps, 0.4)
    vbef2, vaft2 = _impulse(B, fps, 0.2)
    imp4 = np.linalg.norm(vaft4 - vbef4, axis=1)
    imp2 = np.linalg.norm(vaft2 - vbef2, axis=1)
    half = int(np.ceil(fps / physics.RATE / 2)) if fps > physics.RATE else 0
    stack = lambda x: np.nanmax(np.stack([x[np.clip(near + o, 0, T - 1)]
                                          for o in range(-half, half + 1)]), axis=0)
    u = vaft4 / np.maximum(np.linalg.norm(vaft4, axis=1, keepdims=True), 1e-6)

    vis = np.isfinite(B[:, 0])
    dd = np.diff(np.r_[1, vis.astype(np.int8), 1])
    gpos = np.zeros(T)
    glen = np.zeros(T)
    for a, b in zip(np.flatnonzero(dd == -1), np.flatnonzero(dd == 1)):
        gpos[a:b] = (np.arange(b - a) + 0.5) / (b - a)
        glen[a:b] = np.log1p((b - a) / fps)

    s = sgn[near]
    bx, by = B[near, 0] * s, B[near, 1] * s
    beyond = np.max(np.abs(B[near]) - np.array(m.pitch) / 2, axis=1)
    ball = np.column_stack([
        bx / SCALE_POS[0], by / SCALE_POS[1], vb[near, 0] * s / 10, vb[near, 1] * s / 10,
        vis[near].astype(float), stack(imp2) / 10, stack(imp4) / 10, beyond / 5,
        gpos[near], glen[near]])
    ball = np.clip(np.where(np.isfinite(ball), ball, 0.0), -CLIP, CLIP)

    players = np.zeros((n, 2, MAX_PER_TEAM, N_PLAYER_FEATS))
    dref = np.full((n, 2, MAX_PER_TEAM), 99.0)
    R = reference_point(m)[near]
    for t_i, P in enumerate((P_home, P_away)):
        V = np.full_like(P, np.nan)
        V[1:-1] = (P[2:] - P[:-2]) * fps / 2
        Pn, Vn = P[near], V[near]
        rel = B[near][:, None] - Pn
        d = np.linalg.norm(rel, axis=2)
        behind = np.einsum("tnc,tc->tn", rel, u[near]) / np.maximum(d, 1e-6)
        vproj = np.einsum("tnc,tc->tn", Vn, u[near])
        vrel = np.linalg.norm(vb[near][:, None] - Vn, axis=2)
        present = np.isfinite(Pn[..., 0])
        dr = np.linalg.norm(Pn - R[:, None], axis=2)
        key = np.where(present, np.where(np.isfinite(dr), dr, 1e3), np.inf)
        order = np.argsort(key, axis=1, kind="stable")[:, :MAX_PER_TEAM]  # ties: same order on every CPU
        take = lambda x: np.take_along_axis(x, order, axis=1)
        feats = np.stack([
            take(Pn[..., 0]) * s[:, None] / SCALE_POS[0], take(Pn[..., 1]) * s[:, None] / SCALE_POS[1],
            take(Vn[..., 0]) * s[:, None] / 8, take(Vn[..., 1]) * s[:, None] / 8,
            np.where(np.isfinite(take(d)), np.minimum(take(d), 30) / 10, 3.0),
            take(behind), take(vproj) / 8, np.minimum(take(vrel), 30) / 10,
            take(present).astype(float), np.zeros_like(take(d))], axis=2)
        feats[..., 9] = t_i  # team flag (home 0, away 1); swap augmentation exchanges teams
        k = order.shape[1]
        players[:, t_i, :k] = np.clip(np.where(np.isfinite(feats), feats, 0.0), -CLIP, CLIP)
        dk = take(np.where(present, dr, np.nan))
        dref[:, t_i, :k] = np.where(np.isfinite(dk), np.minimum(dk / 10, 98.0), 99.0)
    return players.astype(np.float16), ball.astype(np.float16), near, dref.astype(np.float16)


# Params
N_EXTRA_PLAYER_FEATS = 4
REACH_V = 7.0  # m/s, as physics.V_REACH (see its note)
REACH_STEPS_S = np.arange(0.2, 3.01, 0.2)


def extra_player_features(m, players, dref, near):
    """(T10, 2, MAX, 4) more per-player features in the same slot order as `players`:
    pressure (distance to the nearest opponent, /10 m), heading towards the ball (cosine of
    velocity and direction to the ball), time to reach the ball's current path (/3 s, 1 when
    out of reach or the ball is missing) and the distance to the reference point (/10 m)."""
    n = len(near)
    sgn = physics.attack_sign(m)[near][:, None, None]
    pos = players[..., 0:2].astype(np.float64) * np.array(SCALE_POS) * sgn[..., None]
    vel = players[..., 2:4].astype(np.float64) * 8 * sgn[..., None]
    present = players[..., 8] > 0.5
    pos = np.where(present[..., None], pos, np.nan)
    B = m.ball[near, :2].astype(np.float64)
    vb = np.full_like(m.ball[:, :2], np.nan, dtype=np.float64)
    vb[1:-1] = (m.ball[2:, :2] - m.ball[:-2, :2]) * m.fps / 2
    vb = vb[near]
    out = np.zeros((n, 2, MAX_PER_TEAM, N_EXTRA_PLAYER_FEATS))
    for t in range(2):
        d = np.linalg.norm(pos[:, t, :, None] - pos[:, 1 - t, None], axis=3)
        nearest = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=2)
        out[:, t, :, 0] = np.where(np.isfinite(nearest), np.minimum(nearest, 30) / 10, 3.0)
    to_ball = B[:, None, None] - pos
    dist = np.linalg.norm(to_ball, axis=3)
    sp = np.linalg.norm(vel, axis=3)
    out[..., 1] = np.einsum("tsmc,tsmc->tsm", vel, to_ball) / np.maximum(sp * dist, 1e-6)
    reach = np.full(dist.shape, np.inf)
    for s in REACH_STEPS_S:
        path = (B + vb * s)[:, None, None]
        ok = np.linalg.norm(pos - path, axis=3) <= REACH_V * s + 1.0
        reach = np.where(ok & ~np.isfinite(reach), s, reach)
    out[..., 2] = np.where(np.isfinite(reach), reach / 3, 1.0)
    out[..., 3] = np.minimum(dref.astype(np.float64), 3.0)
    out = np.where(present[..., None] & np.isfinite(out), out, 0.0)
    return np.clip(out, -CLIP, CLIP).astype(np.float16)
