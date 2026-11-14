"""Player-ball interaction physics, computed at the source's native rate (25 fps for the provided
match and DFL, 10 fps for SkillCorner) and then aggregated to the 10 Hz working rate.

Only positions are used (ball x, y and player x, y), with the same code for every source, so
nothing a source has and the provided match lacks can leak in.

Per frame and per team (home = 0, away = 1):

- who moves the ball: at every frame the ball's velocity is fitted over a short window before
  and after; the change between them is the impulse. Each player is scored by how well they
  explain it: distance to the ball, whether they stand behind it with respect to the outgoing
  direction, and their own speed along that direction;
- everyone nearby, not only the nearest: players within several radii, duels (both teams
  close), who reaches the ball first along its current path;
- ball regime: free flight (no impulse, nobody moving with it), touch (impulse), control (the
  ball moves at the speed of a nearby player);
- ball gaps: when the ball disappears its path is projected forward to find who intercepts it
  first (receiver); when it reappears, backwards to find who was on its path (kicker). Both are
  carried through the gap, together with the gap length and position in the gap.

Constants and their origin:
  IMPULSE_S     window of the before / after velocity fits. Broadcast tracking smooths the ball
                with a Gaussian of about 5 frames at 25 fps (0.2 s: the external calibration on
                SkillCorner, experiments/sim_ext_calibration.md, ball_sigma_full), so an impulse
                is spread over roughly +-0.4 s; two scales are kept and the model chooses.
  RADII_M       1 m is the possession zone of Vidal-Codina et al. (2022) and Link & Hoernig (2017);
                2 m their duel / pressure zone; 5 m the local neighbourhood.
  V_REACH       7.0 m/s, the 99th percentile of player speed at 10 Hz (scripts/derive_constants.py:
                DFL 6.76 and both external sources pooled 6.51, 7 when rounded to the whole m/s).
  PATH_S        how far ahead (or back) the ball path is followed: 3 s, the median gap of a pass
                in open play is well below that at 15-25 m/s over 20-40 m.
"""

import numpy as np

from .kin import RATE

# Params
IMPULSE_S = (0.2, 0.4)
RADII_M = (1.0, 2.0, 5.0)
V_REACH = 7.0
PATH_S = 3.0
PATH_STEP_S = 0.2


def _players(m):
    P = np.concatenate([m.home, m.away], axis=1)
    side = np.r_[np.zeros(m.home.shape[1], int), np.ones(m.away.shape[1], int)]
    return P, side


def attack_sign(m):
    """+1 / -1 per frame so that x * sign has the home team attacking +x in every period.

    Label-free: in each period the home team's mean x is compared with the away team's (a team's
    centroid lies on its own half over a whole period)."""
    sign = np.ones(m.n_frames, np.float32)
    for per in np.unique(m.period[m.period > 0]):
        f = m.period == per
        hx = np.nanmean(m.home[f, :, 0])
        ax = np.nanmean(m.away[f, :, 0])
        sign[f] = 1.0 if hx < ax else -1.0
    return sign


def _window_velocity(B, k):
    """(before, after) velocity fits in m/frame over k+1 frames ending / starting at t."""
    T = len(B)
    lag = np.arange(k + 1, dtype=np.float64)
    lag -= lag.mean()
    w = lag / (lag ** 2).sum()
    before = np.zeros((T, 2))
    after = np.zeros((T, 2))
    for j in range(k + 1):
        sb = np.full((T, 2), np.nan)
        sb[k - j:] = B[:T - (k - j)]  # frame t - (k - j)
        before += w[j] * sb
        sa = np.full((T, 2), np.nan)
        sa[:T - j] = B[j:]  # frame t + j
        after += w[j] * sa
    return before, after


def frame_features(m):
    """(T, F) float32 physics features at the native rate and their names."""
    P, side = _players(m)
    B = m.ball[:, :2].astype(np.float64)
    T, N = P.shape[:2]
    fps = m.fps
    sgn = attack_sign(m)
    feats, names = [], []

    def add(name, x):
        feats.append(np.asarray(x, np.float32))
        names.append(name)

    vis = np.isfinite(B[:, 0])
    V = np.full_like(P, np.nan)
    V[1:-1] = (P[2:] - P[:-2]) * fps / 2
    vb = np.full_like(B, np.nan)
    vb[1:-1] = (B[2:] - B[:-2]) * fps / 2

    rel = B[:, None] - P  # player -> ball
    d = np.linalg.norm(rel, axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    teams = [side == 0, side == 1]

    # ball state
    add("ball visible", vis)
    add("ball speed", np.linalg.norm(vb, axis=1))
    half = np.array(m.pitch) / 2
    add("ball beyond line m", np.max(np.abs(B) - half, axis=1))
    add("ball x attack", B[:, 0] * sgn)

    # impulses and who explains them, at two scales
    for s in IMPULSE_S:
        k = max(2, int(round(s * fps)))
        vbef, vaft = _window_velocity(B, k)
        dv = (vaft - vbef) * fps
        imp = np.linalg.norm(dv, axis=1)
        sp_out = np.linalg.norm(vaft, axis=1) * fps
        u = vaft / np.maximum(np.linalg.norm(vaft, axis=1, keepdims=True), 1e-6)
        behind = np.einsum("tnc,tc->tn", rel, u) / np.maximum(d, 1e-6)
        vproj = np.einsum("tnc,tc->tn", V, u)
        add(f"impulse {s}s", imp)
        add(f"turn {s}s", np.einsum("tc,tc->t", vbef, vaft) /
            np.maximum(np.linalg.norm(vbef, axis=1) * np.linalg.norm(vaft, axis=1), 1e-9))
        add(f"speed out {s}s", sp_out)
        for t_i, tm in enumerate(teams):
            # the team's best candidate kicker: nearest player standing behind the ball
            db = np.where((behind > 0) & tm[None], d, np.inf)
            j = db.argmin(1)
            dbest = db[np.arange(T), j]
            add(f"kicker dist {s}s t{t_i}", np.where(np.isfinite(dbest), dbest, np.nan))
            add(f"kicker vproj {s}s t{t_i}",
                np.where(np.isfinite(dbest), vproj[np.arange(T), j], np.nan))

    # everyone nearby
    vrel = np.linalg.norm(vb[:, None] - V, axis=2)
    for t_i, tm in enumerate(teams):
        dt = np.where(tm[None], d, np.inf)
        ds = np.sort(dt, axis=1)
        add(f"nearest t{t_i}", np.where(np.isfinite(ds[:, 0]), ds[:, 0], np.nan))
        add(f"second t{t_i}", np.where(np.isfinite(ds[:, 1]), ds[:, 1], np.nan))
        for r in RADII_M:
            add(f"n within {r:g}m t{t_i}", np.where(vis, (dt < r).sum(1), np.nan))
        # control: ball moving with a nearby player of this team
        vr = np.where(tm[None] & (d < RADII_M[1]), vrel, np.inf).min(1)
        add(f"control relspeed t{t_i}", np.where(np.isfinite(vr), vr, np.nan))
        # closing speed of the nearest player
        j = dt.argmin(1)
        closing = -np.einsum("tc,tc->t", V[np.arange(T), j] - vb, -rel[np.arange(T), j]) / \
            np.maximum(d[np.arange(T), j], 1e-6)
        add(f"closing nearest t{t_i}", np.where(vis, closing, np.nan))

    # who reaches the ball first along its current path (ball at constant velocity)
    steps = np.arange(PATH_STEP_S, PATH_S + 1e-9, PATH_STEP_S)
    arrive = np.full((T, 2), np.inf)
    for s in steps:
        path = B + vb * s
        reach = np.linalg.norm(P - path[:, None], axis=2) <= V_REACH * s + RADII_M[0]
        for t_i, tm in enumerate(teams):
            hit = (reach & tm[None]).any(1) & ~np.isfinite(arrive[:, t_i])
            arrive[hit, t_i] = s
    arrive = np.where(np.isfinite(arrive), arrive, PATH_S + PATH_STEP_S)
    arrive[~vis] = np.nan
    add("arrive first t0", arrive[:, 0])
    add("arrive first t1", arrive[:, 1])

    # ball gaps: receiver from the forward path at the last visible frame, kicker from the
    # backward path at the first visible frame, carried through the gap
    gap_len = np.full(T, np.nan)
    gap_pos = np.full(T, np.nan)
    recv = np.full((T, 2), np.nan)
    kick = np.full((T, 2), np.nan)
    dd = np.diff(np.r_[1, vis.astype(np.int8), 1])
    starts, ends = np.flatnonzero(dd == -1), np.flatnonzero(dd == 1)
    nsteps = len(steps)
    for a, b in zip(starts, ends):
        L = b - a
        gap_len[a:b] = L / fps
        gap_pos[a:b] = (np.arange(L) + 0.5) / L
        if a >= 2 and np.isfinite(vb[a - 2, 0]):
            t0 = a - 1
            v0 = (B[t0] - B[t0 - 1]) * fps if np.isfinite(B[t0 - 1, 0]) else vb[t0 - 1]
            for t_i, tm in enumerate(teams):
                best = PATH_S + PATH_STEP_S
                for s in steps:
                    tt = min(T - 1, t0 + int(round(s * fps)))
                    gap_d = np.linalg.norm(P[tt, tm] - (B[t0] + v0 * s), axis=1)
                    if np.any(gap_d <= RADII_M[1]):
                        best = s
                        break
                recv[a:b, t_i] = best
        if b + 1 < T and np.isfinite(B[b + 1, 0]):
            v1 = (B[b + 1] - B[b]) * fps
            for t_i, tm in enumerate(teams):
                best = np.inf
                for s in steps[:nsteps]:
                    tt = b - int(round(s * fps))
                    if tt < 0:
                        break
                    gap_d = np.linalg.norm(P[tt, tm] - (B[b] - v1 * s), axis=1)
                    if np.isfinite(gap_d).any():
                        best = min(best, np.nanmin(gap_d))
                kick[a:b, t_i] = best if np.isfinite(best) else np.nan
    add("gap length s", gap_len)
    add("gap position", gap_pos)
    add("gap receiver time t0", recv[:, 0])
    add("gap receiver time t1", recv[:, 1])
    add("gap kicker dist t0", kick[:, 0])
    add("gap kicker dist t1", kick[:, 1])

    # scene (players only; available in every frame)
    for t_i, tm in enumerate(teams):
        add(f"team speed t{t_i}", np.nanmean(np.linalg.norm(V[:, tm], axis=2), axis=1))
        add(f"team x attack t{t_i}", np.nanmean(P[:, tm, 0], axis=1) * sgn)
        add(f"team spread t{t_i}", np.nanmean(np.nanstd(P[:, tm], axis=1), axis=1))
        add(f"tracked t{t_i}", np.isfinite(P[:, tm, 0]).sum(1))
    return np.column_stack(feats), names


# Features that mark an instant (an impulse); max-pooled over the 10 Hz bin so a peak between two
# samples is not lost. Everything else is taken at the sample time.
PEAK_PREFIXES = ("impulse", "turn")


def to_rate(X, names, fps, n_out=None, rate=RATE):
    """Aggregate native-rate features to `rate` Hz: max over the bin for impulse features, the
    value at the nearest frame otherwise. Returns (T_out, F) and the source frame of each row."""
    T = len(X)
    if rate >= fps:
        return X.copy(), np.arange(T)
    t = np.arange(0, T - 1, fps / rate)
    near = np.round(t).astype(int)
    out = X[near].copy()
    if fps > rate:
        half = int(np.ceil(fps / rate / 2))
        peak = [i for i, n in enumerate(names) if n.startswith(PEAK_PREFIXES)]
        for i in peak:
            col = X[:, i]
            stack = np.stack([col[np.clip(near + o, 0, T - 1)] for o in range(-half, half + 1)])
            if names[i].startswith("turn"):
                out[:, i] = np.nanmin(stack, axis=0)  # sharpest turn = lowest cosine
            else:
                out[:, i] = np.nanmax(stack, axis=0)
    return out, near
