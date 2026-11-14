"""Degrade clean tracking (DFL, 25 fps) into tracking with the defects of a target tracking.

Every quantity is calibrated on the target WITHOUT labels: what is observed there is whether the
ball and each track exist and how they move. The target of the submission is external broadcast
tracking (SkillCorner, through src/poss/sim_external.py); the provided match is never used. Four
defects are reproduced, in the order a tracker would produce them:

1. Smoothing. A Gaussian filter per track; the player width is fitted by matching label-free
   statistics (second differences, accelerations) between the target and filtered DFL. The ball
   also has a smooth noise floor, reproduced as white noise passed through the same filter. Ball
   width, noise and the snap drift length are fitted together on the full simulation (see
   `calibrate`), because what the target shows is the ball that survives the gaps.
2. Track fragmentation. P(track dies | player context) is a classifier fitted on the target, where
   deaths are observed; the hazard scale and the mean absence are fitted so that the number of
   tracked players per 30 s window has the target's median mean and spread. Each new fragment
   gets a new track id, so identities are not stable.
3. Position snaps (steps faster than a human can run): rare, bursty and never reverting.
   Reproduced non-parametrically from the target's per-window counts, per-frame group sizes and
   magnitudes; a snap corrects a drift that grew linearly since the start of the track piece.
4. Ball gaps. A discrete-time hazard model: P(gap starts | context) on frames with a visible
   ball, P(gap ends | context, time already missing) on missing frames, both classifiers fitted
   on the target, where the target (ball missing) is observed. What the target cannot show is
   the hidden ball during a gap; a tracker re-acquires the ball mostly once it moves again. The
   end hazard is therefore reweighted by a density ratio over bins of the reappearing ball
   (speed, beyond a line, distance to the nearest player), iterated until the simulated
   reappearances have the target's composition; the onset hazard is reweighted the same way.

`degrade(..., spread=s)` multiplies every rate and scale by exp(U(-s, s)) (domain randomisation).
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .kin import velocity
from .match import Match

# Params
V_MAX = 12.4  # m/s, top human sprint speed; a faster frame-to-frame step is a tracking artefact
# Elapsed-time bins (s) for the gap-end hazard; geometric, covering gaps from 0.1 s to minutes.
ELAPSED_EDGES = np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 64, np.inf])
WINDOW_S = 30  # window for the per-window statistics (snap counts, tracked players)
BALL_SIGMAS = np.arange(0, 12.5, 0.5)  # frames at 25 fps, searched for the ball filter
PLAYER_SIGMAS = np.arange(0, 6.25, 0.25)  # frames at 25 fps, searched for the player filter
# Joint full-simulation search for the ball filter, its noise floor (m) and the snap drift (s)
BALL_FIT = [(sb, nz, rp) for sb in (4.0, 5.0, 6.0) for nz in (0.0, 0.03, 0.1, 0.3)
            for rp in (0.5, np.inf)]
# Bins of the reappearing ball for the gap-end reweighting (speed m/s, nearest player m)
REAPP_SPEED = np.array([0.5, 2, 5, 10, 15])
REAPP_NEAR = np.array([1, 3, 10])
REAPP_LAG = 2  # frames after reappearance at which the ball is described (speed needs neighbours)
REAPP_ITERS = 5
DEATH_SCALES = np.array([0.5, 0.71, 1.0, 1.41, 2.0])  # multipliers of the fitted death hazard
ABSENCE_S = np.array([1, 2, 4, 8, 16, 32])  # candidate mean absences (s)
LGB = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200,
           subsample=0.5, subsample_freq=1, colsample_bytree=0.8, deterministic=True,
           num_threads=4, verbose=-1, random_state=0)


def _players(m):
    P = np.concatenate([m.home, m.away], axis=1)
    side = np.r_[np.zeros(m.home.shape[1], int), np.ones(m.away.shape[1], int)]
    return P, side


def runs(mask, min_len=1):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
            if b - a >= min_len]


def smooth_runs(x, sigma):
    """Gaussian filter along time inside each finite run of x (T, 2); NaN stays NaN."""
    if sigma <= 0:
        return x
    y = x.copy()
    for a, b in runs(np.isfinite(x[:, 0])):
        y[a:b] = gaussian_filter1d(x[a:b], sigma, axis=0, mode="nearest")
    return y


def roughness(x, mask, fps, min_len):
    """Label-free smoothness statistics of a trajectory: second-difference magnitude
    percentiles (as accelerations) and the lag-1 autocorrelation of second differences."""
    acc, num, den = [], 0.0, 0.0
    for a, b in runs(mask & np.isfinite(x[:, 0]), min_len):
        dd = np.diff(x[a:b], 2, axis=0)
        acc.append(np.linalg.norm(dd, axis=1) * fps ** 2)
        num += (dd[1:] * dd[:-1]).sum()
        den += (dd ** 2).sum()
    acc = np.concatenate(acc)
    return np.r_[np.log(np.percentile(acc, [50, 90, 99]) + 1e-3), num / den]


def scene_context(m):
    """Per-frame features from the players only (available in every source and in every frame)."""
    P, side = _players(m)
    sp = np.linalg.norm(velocity(P, m.fps), axis=2)
    sp_mean = np.nanmean(sp, axis=1)
    lag = int(m.fps)
    trend = np.full_like(sp_mean, np.nan)
    trend[lag:] = sp_mean[lag:] - sp_mean[:-lag]
    half_l = m.pitch[0] / 2
    cx = np.nanmean(P[..., 0], axis=1)
    spread = np.nanmean(np.nanstd(P, axis=1), axis=1)
    n = np.isfinite(P[..., 0]).sum(1)
    fast = (sp > 4).sum(1)
    return np.column_stack([sp_mean, trend, half_l - np.abs(cx), spread, n, fast,
                            np.nanpercentile(sp, 90, axis=1)]).astype(np.float32)


def ball_context(m):
    """Per-frame features of the ball where it is visible (NaN elsewhere)."""
    P, _ = _players(m)
    B = m.ball[:, :2]
    vb = np.linalg.norm(velocity(B, m.fps), axis=1)
    half_l, half_w = m.pitch[0] / 2, m.pitch[1] / 2
    out_x = np.abs(B[:, 0]) - half_l
    out_y = np.abs(B[:, 1]) - half_w
    d = np.linalg.norm(P - B[:, None], axis=2)
    dmin = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    dmin[~np.isfinite(dmin)] = np.nan
    n5 = (d < 5).sum(1).astype(np.float32)
    n5[np.isnan(B[:, 0])] = np.nan
    return np.column_stack([vb, out_x, out_y, np.maximum(out_x, out_y), dmin, n5]).astype(np.float32)


def track_context(m):
    """Per player-frame features for the track-death hazard: (T, N, F)."""
    P, side = _players(m)
    sp = np.linalg.norm(velocity(P, m.fps), axis=2)
    T, N = P.shape[:2]
    near = np.full((T, N), np.nan, np.float32)
    crowd = np.zeros((T, N), np.float32)
    idx = np.arange(N)
    for a in range(0, T, 5000):  # pairwise distances in blocks to bound memory
        D = np.linalg.norm(P[a:a + 5000, :, None] - P[a:a + 5000, None], axis=3)
        D[:, idx, idx] = np.inf
        D = np.where(np.isfinite(D), D, np.inf)
        nb = D.min(2)
        near[a:a + 5000] = np.where(np.isfinite(nb), nb, np.nan)
        crowd[a:a + 5000] = (D < 2).sum(2)
    gx = m.pitch[0] / 2 - np.abs(P[..., 0])
    gy = m.pitch[1] / 2 - np.abs(P[..., 1])
    return np.stack([sp, near, crowd, gx, gy], axis=2).astype(np.float32)


def _elapsed_bin(e):
    return np.searchsorted(ELAPSED_EDGES, e, side="right") - 1


def tracked_stats(P, play, fps):
    """Median over full 30 s windows of the mean and std of the number of tracked players."""
    n = np.isfinite(P[..., 0]).sum(1)
    w = int(WINDOW_S * fps)
    mean, std = [], []
    for a in range(0, len(n) - w + 1, w):
        if play[a:a + w].all():
            mean.append(n[a:a + w].mean())
            std.append(n[a:a + w].std())
    return np.array([np.median(mean), np.median(std)])


def death_prob(m, cal):
    """P(track dies at t) for every player-frame where the player exists (0 elsewhere)."""
    P, _ = _players(m)
    exist = np.isfinite(P[..., 0])
    p = np.zeros(exist.shape)
    p[exist] = cal.death.predict_proba(track_context(m)[exist])[:, 1]
    return p


def fragment(p, play, k_death, absence_mean_s, fps, rng):
    """Track ids (T, N) after random deaths with hazard k_death * p and exponential absences;
    -1 while absent, a fresh id for every new fragment."""
    T, N = p.shape
    dead = (p > 0) & play[:, None] & (rng.random((T, N)) < p * k_death)
    track = np.tile(np.arange(N), (T, 1))
    next_id = N
    for i in range(N):
        for t in np.flatnonzero(dead[:, i]):
            if track[t, i] < 0:
                continue  # already absent
            gap = int(np.ceil(rng.exponential(absence_mean_s) * fps))
            track[t + 1:t + 1 + gap, i] = -1
            if t + 1 + gap < T:
                track[t + 1 + gap:, i] = next_id
                next_id += 1
    return track


def visible_ball_stats(m):
    """Label-free statistics of the ball where it is visible: share within 1 m of a player,
    median nearest-player distance, median and p90 acceleration."""
    P, _ = _players(m)
    B = m.ball[:, :2]
    vis = np.isfinite(B[:, 0]) & (m.period > 0)
    d = np.linalg.norm(P - B[:, None], axis=2)
    d = np.where(np.isfinite(d), d, np.inf).min(1)
    a = np.r_[np.nan, np.linalg.norm(np.diff(B, 2, axis=0), axis=1) * m.fps ** 2, np.nan]
    return np.array([(d[vis] < 1).mean(), np.median(d[vis]), np.nanmedian(a[vis]),
                     np.nanpercentile(a[vis], 90)])


def reapp_bin(bctx):
    """Bin index of ball-context rows (see ball_context): speed x beyond-line x nearest player."""
    sp = np.searchsorted(REAPP_SPEED, np.nan_to_num(bctx[:, 0], nan=0.0))
    out = (np.nan_to_num(bctx[:, 3], nan=0.0) > 0).astype(int)
    nr = np.searchsorted(REAPP_NEAR, np.nan_to_num(bctx[:, 4], nan=99.0))
    return (sp * 2 + out) * (len(REAPP_NEAR) + 1) + nr


N_REAPP_BINS = (len(REAPP_SPEED) + 1) * 2 * (len(REAPP_NEAR) + 1)


def reappearance_hist(m):
    """Counts per bin of the ball REAPP_LAG frames after each reappearance in play."""
    vis = np.isfinite(m.ball[:, 0])
    d = np.diff(np.r_[1, vis.astype(np.int8), 1])
    ends = np.flatnonzero(d == 1)
    ends = ends[(ends > 0) & (ends < len(vis) - REAPP_LAG)]
    ends = ends[m.period[ends] > 0]
    ctx = ball_context(m)[ends + REAPP_LAG]
    return np.bincount(reapp_bin(ctx), minlength=N_REAPP_BINS).astype(float)


def visible_hist(m):
    """Counts per bin (see reapp_bin) of the visible ball in play."""
    vis = np.isfinite(m.ball[:, 0]) & (m.period > 0)
    return np.bincount(reapp_bin(ball_context(m)[vis]), minlength=N_REAPP_BINS).astype(float)


def visible_share(m):
    return float(np.isfinite(m.ball[m.period > 0, 0]).mean())


def fit_visibility(cal, bar, refs, iters=6):
    """Global onset scale so the simulated ball is visible on the same share of graded frames as
    the target's (bisection on log scale; the reweightings above fix the composition, not the level)."""
    target = visible_share(bar)
    lo, hi = -2.0, 2.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        cal.onset_scale = float(np.exp(mid))
        v = np.mean([visible_share(degrade(r, cal, np.random.default_rng(0))) for r in refs])
        if v > target:
            lo = mid  # too visible: start gaps more often
        else:
            hi = mid
    cal.onset_scale = float(np.exp((lo + hi) / 2))
    cal.stats.update(visible_target=target, onset_scale=cal.onset_scale)
    return cal


@dataclass
class Calibration:
    fps: float
    ball_sigma: float
    player_sigma: float
    ball_on: object
    ball_off: object
    death: object
    death_scale: float
    absence_mean_s: float
    snaps_per_window: np.ndarray
    snap_group_sizes: np.ndarray
    snap_magnitudes: np.ndarray
    snap_ramp_s: float = np.inf
    ball_noise: float = 0.0
    reapp_ratio: np.ndarray = None
    onset_ratio: np.ndarray = None
    onset_scale: float = 1.0
    stats: dict = field(default_factory=dict)


def calibrate(target, refs):
    """Fit every simulator component.

    `target`: the tracking to imitate, at 25 fps (external: sim_external.target); only
    label-free statistics are taken.
    `refs`: clean external matches at the same rate, used to find the filter widths and the
    absence parameters whose simulated statistics match the target's.
    """
    bar = target
    fps = bar.fps
    play = bar.period > 0
    P, _ = _players(bar)
    stats = {}

    # 1. smoothing widths by matching roughness statistics
    def fit_sigma(target, grid, get, min_len):
        best, err = 0.0, np.inf
        for s in grid:
            vals = np.mean([roughness(smooth_runs(x, s), msk, fps, min_len)
                            for r in refs for x, msk in get(r)], axis=0)
            e = np.sum((vals - target) ** 2)
            if e < err:
                best, err = s, e
        return float(best), err

    t_ball = roughness(bar.ball[:, :2], play, fps, 100)
    ball_sigma, eb = fit_sigma(t_ball, BALL_SIGMAS,
                               lambda r: [(r.ball[:, :2], r.alive == 1)], 100)
    def busiest(Q, k=8):  # a few well-covered tracks are enough for the player statistics
        return np.argsort(-np.isfinite(Q[..., 0]).sum(0))[:k]

    t_play = np.mean([roughness(P[:, i], play, fps, 250) for i in busiest(P)], axis=0)
    player_sigma, ep = fit_sigma(
        t_play, PLAYER_SIGMAS,
        lambda r: [(_players(r)[0][:, i], r.period > 0) for i in busiest(_players(r)[0])], 250)
    stats.update(ball_roughness=t_ball.round(3).tolist(), ball_sigma=ball_sigma,
                 player_roughness=t_play.round(3).tolist(), player_sigma=player_sigma)

    # 2. track deaths: hazard classifier on the target, then scale + absence by moment matching
    miss = np.isnan(bar.ball[:, 0])
    nxt = np.r_[miss[1:], True]
    same = np.r_[bar.period[1:] == bar.period[:-1], False] & play
    fin = np.isfinite(P[..., 0]) & play[:, None]
    dies = fin & ~np.r_[fin[1:], np.zeros((1, fin.shape[1]), bool)]
    rows = fin & same[:, None]
    tc = track_context(bar)
    rng = np.random.default_rng(0)
    keep = rows & (rng.random(rows.shape) < 0.25)
    death = lgb.LGBMClassifier(**LGB).fit(tc[keep], dies[keep])
    t_tracked = tracked_stats(P, play, fps)

    # 3. snaps: empirical per-window counts, per-frame group sizes and magnitudes
    step = np.linalg.norm(np.diff(P, axis=0), axis=2)
    ok = rows[:-1] & np.isfinite(step)
    S = np.argwhere(ok & (step > V_MAX / fps))
    win = int(WINDOW_S * fps)
    n_win = len(P) // win
    full = np.array([play[i * win:(i + 1) * win].all() for i in range(n_win)])
    per_window = np.bincount(S[:, 0] // win, minlength=n_win)[:n_win][full]
    per_frame = np.bincount(S[:, 0])
    group_sizes = per_frame[per_frame > 0]
    magnitudes = step[S[:, 0], S[:, 1]]

    # 4. ball gaps: onset on visible frames, end on missing frames given elapsed time
    scene = scene_context(bar)
    ball = ball_context(bar)
    vis = same & ~miss
    on = lgb.LGBMClassifier(**LGB).fit(np.hstack([scene, ball])[vis], nxt[vis])
    elapsed = np.zeros(len(miss))
    run = 0
    for t in range(len(miss)):
        run = run + 1 if miss[t] and play[t] else 0
        elapsed[t] = run / fps
    last = ball.copy()
    for t in range(1, len(last)):  # last visible ball context carried through a gap
        if miss[t]:
            last[t] = last[t - 1]
    gap = same & miss
    Xoff = np.hstack([scene, last, _elapsed_bin(elapsed)[:, None]])
    off = lgb.LGBMClassifier(**LGB).fit(Xoff[gap], ~nxt[gap])

    cal = Calibration(fps=fps, ball_sigma=ball_sigma, player_sigma=player_sigma, ball_on=on,
                      ball_off=off, death=death, death_scale=1.0, absence_mean_s=1.0,
                      snaps_per_window=per_window, snap_group_sizes=group_sizes,
                      snap_magnitudes=magnitudes, stats=stats)
    smoothed = [degrade(r, cal, np.random.default_rng(0), parts=("smooth",)) for r in refs]
    probs = [death_prob(r, cal) for r in smoothed]
    best, err = None, np.inf
    for k in DEATH_SCALES:
        for a in ABSENCE_S:
            sim = []
            for r, p in zip(smoothed, probs):
                tr = fragment(p, r.period > 0, k, a, fps, np.random.default_rng(0))
                Pr = _players(r)[0].copy()
                Pr[tr < 0] = np.nan
                sim.append(tracked_stats(Pr, r.period > 0, fps))
            sim = np.mean(sim, axis=0)
            e = np.sum(((sim - t_tracked) / t_tracked) ** 2)
            if e < err:
                best, err = (float(k), float(a), sim), e
    cal.death_scale, cal.absence_mean_s = best[0], best[1]

    # 4b. gap-end reweighting so simulated reappearances look like the target's
    t_reapp = reappearance_hist(bar)

    t_vis_hist = visible_hist(bar)

    def norm(c):
        return (c + 1) / (c.sum() + N_REAPP_BINS)

    def fit_reapp(iters):
        if cal.reapp_ratio is None:
            cal.reapp_ratio = np.ones(N_REAPP_BINS)
            cal.onset_ratio = np.ones(N_REAPP_BINS)
        for _ in range(iters):
            sims = [degrade(r, cal, np.random.default_rng(0)) for r in refs]
            h = sum(reappearance_hist(x) for x in sims)
            v = sum(visible_hist(x) for x in sims)
            cal.reapp_ratio = np.clip(cal.reapp_ratio * norm(t_reapp) / norm(h), 1e-3, 1e3)
            cal.onset_ratio = np.clip(cal.onset_ratio * norm(v) / norm(t_vis_hist), 1e-3, 1e3)
        return h

    fit_reapp(REAPP_ITERS)

    # 5. ball filter, noise floor and snap drift, on the full simulation
    t_vis = visible_ball_stats(bar)
    best5, err5 = None, np.inf
    for sb, nz, rp in BALL_FIT:
        cal.ball_sigma, cal.ball_noise, cal.snap_ramp_s = sb, nz, rp
        v = np.mean([visible_ball_stats(degrade(r, cal, np.random.default_rng(0)))
                     for r in refs], axis=0)
        e = np.sum(((v - t_vis) / t_vis) ** 2)
        if e < err5:
            best5, err5 = (sb, nz, rp, v), e
    cal.ball_sigma, cal.ball_noise, cal.snap_ramp_s = best5[:3]
    h = fit_reapp(1)
    fit_visibility(cal, bar, refs)
    per_speed = lambda c: c.reshape(len(REAPP_SPEED) + 1, -1).sum(1) / c.sum()
    stats.update(reapp_speed_bins_bar=per_speed(t_reapp).round(3).tolist(),
                 reapp_speed_bins_sim=per_speed(h).round(3).tolist())
    stats.update(visible_ball_target=t_vis.round(3).tolist(),
                 visible_ball_sim=best5[3].round(3).tolist(), ball_sigma_full=best5[0],
                 ball_noise=best5[1], snap_ramp_s=best5[2])
    stats.update(ball_missing=float(miss[play].mean()), tracked_target=t_tracked.round(3).tolist(),
                 tracked_sim=best[2].round(3).tolist(), death_scale=best[0],
                 absence_mean_s=best[1], snaps=int(len(S)),
                 snaps_zero_windows=float((per_window == 0).mean()),
                 snap_median_m=float(np.median(magnitudes)))
    return cal


def degrade(m, cal, rng, spread=0.0, parts=("smooth", "players", "snaps", "ball")):
    """Copy of the clean match `m` (at cal.fps) with the target's defects. Labels are untouched."""
    assert m.fps == cal.fps
    jitter = (lambda: np.exp(rng.uniform(-spread, spread))) if spread else (lambda: 1.0)
    P, side = _players(m)
    P = P.copy()
    ball = m.ball.copy()
    T, N = P.shape[:2]
    play = m.period > 0
    fps = m.fps

    if "smooth" in parts:
        sb, sp = cal.ball_sigma * jitter(), cal.player_sigma * jitter()
        ball[:, :2] = smooth_runs(ball[:, :2], sb)
        if cal.ball_noise > 0:
            white = rng.normal(0, 1, (T, 2))
            col = gaussian_filter1d(white, sb, axis=0)
            ball[:, :2] += col / col.std() * cal.ball_noise * jitter()
        for i in range(N):
            P[:, i] = smooth_runs(P[:, i], sp)

    track = np.tile(np.arange(N), (T, 1))  # track id per player-frame, -1 when absent
    if "players" in parts:
        k_death, k_abs = cal.death_scale * jitter(), cal.absence_mean_s * jitter()
        tmp = Match(m.name, m.source, fps, m.pitch, P[:, side == 0], P[:, side == 1], ball,
                    m.period, m.alive, m.team)
        track = fragment(death_prob(tmp, cal), play, k_death, k_abs, fps, rng)
        P[track < 0] = np.nan

    if "snaps" in parts:
        k_snap, k_mag = jitter(), jitter()
        win = int(WINDOW_S * fps)
        for a in range(0, T - win + 1, win):
            if not play[a:a + win].all():
                continue
            n = int(round(rng.choice(cal.snaps_per_window) * k_snap))
            while n > 0:
                t = a + 1 + rng.integers(win - 1)
                live = np.flatnonzero(np.isfinite(P[t - 1, :, 0]) & np.isfinite(P[t, :, 0]))
                g = min(n, int(rng.choice(cal.snap_group_sizes)), len(live))
                if g == 0:
                    break
                for i in rng.choice(live, g, replace=False):
                    d = rng.choice(cal.snap_magnitudes) * k_mag
                    ang = rng.uniform(0, 2 * np.pi)
                    s = t - 1  # start of this track piece
                    floor = t - 1 - int(cal.snap_ramp_s * fps) if np.isfinite(cal.snap_ramp_s) else 0
                    while s > max(floor, 0) and track[s - 1, i] == track[t - 1, i] and \
                            np.isfinite(P[s - 1, i, 0]):
                        s -= 1
                    ramp = np.linspace(0, 1, t - s + 1)[1:]
                    P[s:t, i] += ramp[:, None] * d * np.array([np.cos(ang), np.sin(ang)])
                n -= g

    if "ball" in parts:
        k_on, k_off = jitter(), jitter()
        tmp = Match(m.name, m.source, fps, m.pitch, P[:, side == 0], P[:, side == 1], ball,
                    m.period, m.alive, m.team)
        scene = scene_context(tmp)
        bctx = ball_context(tmp)
        p_on = cal.ball_on.predict_proba(np.hstack([scene, bctx]))[:, 1] * k_on
        if cal.onset_ratio is not None:
            p_on = p_on * cal.onset_ratio[reapp_bin(bctx)]
        p_on = np.minimum(1.0, p_on * cal.onset_scale)
        u = rng.random(T)
        missing = np.zeros(T, bool)
        t = 0
        while t < T:
            if not (play[t] and u[t] < p_on[t]):
                t += 1
                continue
            # a gap starts at t + 1; the end hazard sees the ball context of frame t (last seen)
            end = t + 1
            while end < T and m.period[end] == m.period[t]:
                end += 1
            span = np.arange(t + 1, end)
            if not len(span):
                t += 1
                continue
            e = _elapsed_bin((span - t) / fps)
            X = np.hstack([scene[span], np.repeat(bctx[t][None], len(span), 0), e[:, None]])
            q = cal.ball_off.predict_proba(X)[:, 1] * k_off
            if cal.reapp_ratio is not None:
                q = q * cal.reapp_ratio[reapp_bin(bctx[np.minimum(span + 1 + REAPP_LAG, T - 1)])]
            q = np.minimum(1.0, q)
            stop = np.flatnonzero(rng.random(len(span)) < q)
            n_miss = stop[0] + 1 if len(stop) else len(span)
            missing[span[:n_miss]] = True
            t = span[n_miss - 1] + 1
        ball[missing] = np.nan

    meta = dict(m.meta)
    meta["track_id"] = track.astype(np.int32)
    return Match(m.name, m.source + "+sim", fps, m.pitch, P[:, side == 0].astype(np.float32),
                 P[:, side == 1].astype(np.float32), ball, m.period, m.alive, m.team, meta)
