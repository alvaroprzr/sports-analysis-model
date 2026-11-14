"""B2 development bench, the same folds, degradations and held-out sets as b1_dev.py.

The model works at 5 Hz (every second 10 Hz row; PathCRF trains at 5 fps and gains only 1.5 F1
from 25 fps at twelve times the cost); its two probabilities are repeated back to the 10 Hz rows
and written in b1_dev's prediction format, so `b1_dev.py decode b2` and b1_compare.py apply.

    python scripts/b2_dev.py [--steps N] [--tag NAME]
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
from poss import b2, dataset, simulate, tokens  # noqa: E402

# Params
SUB = 2  # 10 Hz rows -> 5 Hz
SC_SHARE = 1 / 3  # share of SkillCorner windows per batch (half the weight of DFL)
# Early stopping: the last training DFL match and two training SkillCorner matches are held out
# of training as a validation set (never the test fold); its loss is checked every VAL_EVERY steps
# on VAL_WINDOWS fixed windows; training stops after PATIENCE checks without improvement and the
# best weights are kept. `--steps` is then the maximum.
VAL_EVERY = 100
VAL_WINDOWS = 48
PATIENCE = 4
WARMUP = 100
# Reference configuration; the search (scripts/b2_search.py) overrides any of these
DEFAULT = dict(d=48, hidden=64, heads=4, lr=2e-3, weight_decay=1e-4, win_s=60, batch=8, aux=1.0,
               social="full", k_near=8, temporal="gru",
               boundary_w=1.0, boundary_r_s=1.0, change_head=False, change_w=1.0,
               player_extra=False, labels="flag", use_sc=True,
               ball_head=False, ball_w=1.0, ball_windows=2)
# ball_head (F1): an auxiliary output reconstructs the ball (x, y, attacking frame, scaled as the
# ball token) from the players where it is hidden: on the DFL copies the clean ball where the
# degradation hid it; on extra copies of SkillCorner with more ball gaps (gap_copy),
# the observed ball where the extra gaps hid it (no possession loss on those copies). Huber loss with weight ball_w; ball_windows
# windows of the extra copies per step.
# labels: "flag" = the DFL BallPossession flag; "control" = its control reading (src/poss/control.py,
# data/cache/b1/control/<mid>.npz from scripts/control_1b.py). SkillCorner labels are unchanged.
# use_sc: False trains on DFL only (SkillCorner is still evaluated).
# boundary_w > 1: frames within boundary_r_s of a change of label weigh boundary_w times more in
# the per-frame losses; change_head: an extra output predicts "the label changes within
# boundary_r_s" (auxiliary task, not used by the decoder). Both target the weak point measured
# on B1: short possessions and hand-overs.
N_CTX = 48  # the base physics features of B1 (first columns of the B1 rows)
THREADS = 6
# v2 of the tokens: ordered by reference point; the slot count is part of the cache name
_M = "" if tokens.MAX_PER_TEAM == 14 else f"_m{tokens.MAX_PER_TEAM}"
CACHE = os.path.join(b1_dev.CACHE, "tokens2" + _M)


EXTRA = False  # set per run from cfg["player_extra"]: 4 more per-player features (tokens3)
LABELS = "flag"  # set per run from cfg["labels"]
_CONTROL = {}
CACHE_EXTRA = os.path.join(b1_dev.CACHE, "tokens3" + _M)
MODELS = os.path.join(b1_dev.CACHE, "models")  # weights per fold, for prediction-only reuse


def copy_tokens(key, make_match, rows):
    if EXTRA:
        path = os.path.join(CACHE_EXTRA, key + ".npz")
        if not os.path.exists(path):
            base = _copy_tokens(key, make_match, rows)
            m = make_match()
            near = np.round(np.arange(0, m.n_frames - 1, m.fps / 10)).astype(int)
            extra = tokens.extra_player_features(m, base["players"], base["dref"], near)
            base["players"] = np.concatenate([base["players"], extra], axis=-1)
            os.makedirs(CACHE_EXTRA, exist_ok=True)
            np.savez(path, **base)
        z = np.load(path)
        return {k: z[k] for k in z.files}
    return _copy_tokens(key, make_match, rows)


def _copy_tokens(key, make_match, rows):
    path = os.path.join(CACHE, key + ".npz")
    if not os.path.exists(path):
        pl, bl, near, dref = tokens.tokens(make_match())
        F, _, y, period, near_rows = rows()
        assert np.array_equal(near, near_rows)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, players=pl, ball=bl, dref=dref, ctx=F[:, :N_CTX], y=y, period=period)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def dfl_copy(mid, seed, swap, clean=False):
    tag = "clean" if clean else f"sim{seed}"
    if b1_dev.SIM and not clean:
        tag += f"_{b1_dev.SIM}"

    def make():
        m = b1_dev.load("dfl", mid)
        if not clean:
            spread = 0.0 if seed == b1_dev.TEST_SEED else b1_dev.SPREAD
            m = simulate.degrade(m, b1_dev.copy_calibration(mid, seed), np.random.default_rng(seed),
                                 spread=spread)
        return dataset.swap_teams(m) if swap else m
    c = copy_tokens(f"dfl_{mid}_{tag}_{int(swap)}", make,
                    lambda: b1_dev.dfl_rows(mid, seed, swap, clean=clean))
    if LABELS == "control":
        c["y"] = control_rows(mid, swap, len(c["y"]))
    if BALL and not clean:
        ref = dfl_copy(mid, 0, swap, clean=True)["ball"]
        c["bxy"] = ref[:, :2].astype(np.float32)
        c["bmask"] = (c["ball"][:, 4] < 0.5) & (ref[:, 4] > 0.5)
    return c


BALL = False  # set per run from cfg["ball_head"]
GAP_SHARE, GAP_MEAN_S = 0.25, 3.0  # extra ball gaps: share of the visible frames, mean length


def gap_copy(mid, seed=0):
    """A copy of a SkillCorner match with extra ball gaps; target = the observed ball on the rows
    the extra gaps hid (external data only)."""
    key = f"sc_{mid}_gap{seed}"

    def make():
        m = b1_dev.load("skillcorner", mid)
        rng = np.random.default_rng(seed)
        vis = np.flatnonzero(np.isfinite(m.ball[:, 0]) & (m.period > 0))
        n_gaps = int(GAP_SHARE * len(vis) / (GAP_MEAN_S * m.fps))
        ball = m.ball.copy()
        for st in rng.choice(vis, n_gaps, replace=False):
            ln = int(np.ceil(rng.exponential(GAP_MEAN_S) * m.fps))
            ball[st:st + ln] = np.nan
        m.ball = ball
        return m
    def rows():
        # at 10 fps dataset.rows gives one more final row than tokens.tokens (the cached rows of
        # the other copies were built the same way and trimmed); keep the tokens' rows
        F, cols, y, period, near = dataset.rows(make())
        m = make()
        k = len(np.arange(0, m.n_frames - 1, m.fps / 10))
        return F[:k], cols, y[:k], period[:k], near[:k]
    c = copy_tokens(key, make, rows)
    ref = sc_copy(mid, False)
    assert len(ref["ball"]) == len(c["ball"]), (key, len(ref["ball"]), len(c["ball"]))
    c["bxy"] = ref["ball"][:, :2].astype(np.float32)
    c["bmask"] = (c["ball"][:, 4] < 0.5) & (ref["ball"][:, 4] > 0.5)
    c["y"] = np.full(len(c["bmask"]), dataset.X, np.int8)
    return c


def control_codes(mid):
    """Native-rate codes of a DFL match with the control reading of its team labels."""
    if mid not in _CONTROL:
        m = b1_dev.load("dfl", mid)
        team = np.load(os.path.join(b1_dev.CACHE, "control", f"{mid}.npz"), allow_pickle=True)["team"]
        _CONTROL[mid] = (dataset.codes(m.alive, team, m.period), m.n_frames, m.fps)
    return _CONTROL[mid]


def control_rows(mid, swap, n_rows):
    """Control-reading labels at the 10 Hz rows of a copy (same rows as dataset.rows)."""
    g, n, fps = control_codes(mid)
    near = np.round(np.arange(0, n - 1, fps / 10)).astype(int)
    assert len(near) == n_rows, (mid, len(near), n_rows)
    y = g[near].copy()
    if swap:
        y = np.where(y == dataset.H, dataset.A, np.where(y == dataset.A, dataset.H, y)).astype(y.dtype)
    return y


def sc_copy(mid, swap):
    def make():
        m = b1_dev.load("skillcorner", mid)
        return dataset.swap_teams(m) if swap else m
    return copy_tokens(f"sc_{mid}_{int(swap)}", make, lambda: b1_dev.sc_rows(mid, swap))


def ctx_stats(copies):
    """Median and IQR per context feature over the training copies (label-free)."""
    C = np.vstack([c["ctx"][::50] for c in copies]).astype(np.float64)
    med = np.nanmedian(C, 0)
    iqr = np.nanpercentile(C, 75, 0) - np.nanpercentile(C, 25, 0)
    return med, np.where(iqr > 0, iqr, 1.0)


def norm_ctx(c, stats):
    z = (c.astype(np.float64) - stats[0]) / stats[1]
    return np.clip(np.nan_to_num(z, nan=0.0), -5, 5).astype(np.float32)


def windows(copy, win):
    """Start indices (5 Hz) of the training windows of one copy, inside single periods."""
    period = copy["period"][::SUB]
    starts = []
    for per in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == per)
        starts += list(range(idx[0], idx[-1] - win + 2, win // 2))
    return starts


def batch(copies, picks, stats, win, with_ball=False):
    pl, bl, cx, dr, y, bxy, bm = [], [], [], [], [], [], []
    for ci, s in picks:
        c = copies[ci]
        sl = slice(s * SUB, (s + win) * SUB, SUB)
        pl.append(c["players"][sl])
        bl.append(c["ball"][sl])
        cx.append(norm_ctx(c["ctx"][sl], stats))
        dr.append(c["dref"][sl])
        y.append(c["y"][sl])
        if with_ball:
            bxy.append(c["bxy"][sl])
            bm.append(c["bmask"][sl])
    t = lambda x: b2.to_tensor(np.stack(x))
    out = (t(pl), t(bl), t(cx), t(dr), torch.from_numpy(np.stack(y).astype(np.int64)))
    if with_ball:
        out = out + (t(bxy), torch.from_numpy(np.stack(bm)))
    return out


def loss_ball(lg, bxy, bmask):
    if not bmask.any():
        return lg.sum() * 0
    return torch.nn.functional.smooth_l1_loss(lg[..., -2:][bmask], bxy[bmask], beta=0.1)


def near_change(y, r):
    """(B, T) bool: a change of label within r frames (labels X excluded)."""
    ch = torch.zeros_like(y, dtype=torch.bool)
    ch[:, 1:] = (y[:, 1:] != y[:, :-1]) & (y[:, 1:] < dataset.X) & (y[:, :-1] < dataset.X)
    if r > 0:
        ch = torch.nn.functional.max_pool1d(ch.float()[:, None], 2 * r + 1, 1, r)[:, 0] > 0
    return ch


def loss_dfl(model, pl, bl, cx, dr, y, aux=1.0, cfg=None, bxy=None, bmask=None):
    cfg = cfg or DEFAULT
    lg = model.logits(pl, bl, cx, dr)
    extra = cfg["ball_w"] * loss_ball(lg, bxy, bmask) if bxy is not None else 0.0
    em = model.emissions(lg)
    ok = (y < dataset.X).all(1)
    loss = model.crf_nll(em[ok], y[ok]) if ok.any() else lg.sum() * 0
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    r = int(round(cfg["boundary_r_s"] * 10 / SUB))
    nc = near_change(y, r)
    w = torch.where(nc, torch.tensor(float(cfg["boundary_w"])), torch.tensor(1.0))
    lab = y < dataset.X
    loss = loss + aux * (bce(lg[..., 0][lab], (y[lab] != dataset.D).float(), reduction="none")
                         * w[lab]).sum() / w[lab].sum()
    alive = y < dataset.D
    if alive.any():
        loss = loss + aux * (bce(lg[..., 1][alive], (y[alive] == dataset.H).float(),
                                 reduction="none") * w[alive]).sum() / w[alive].sum()
    if cfg["change_head"]:
        loss = loss + cfg["change_w"] * bce(lg[..., 2][lab], nc[lab].float())
    return loss + extra


def loss_sc(model, pl, bl, cx, dr, y):
    lg = model.logits(pl, bl, cx, dr)
    alive = y < dataset.D
    if not alive.any():
        return lg.sum() * 0
    return torch.nn.functional.binary_cross_entropy_with_logits(
        lg[..., 1][alive], (y[alive] == dataset.H).float())


@torch.no_grad()
def predict(model, copy, stats):
    """p_alive, p_home at the 10 Hz rows (5 Hz outputs repeated)."""
    model.eval()
    n = len(copy["y"])
    pa, ph = np.full(n, 0.5), np.full(n, 0.5)
    period = copy["period"]
    for per in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == per)
        sub = idx[::SUB]
        lg = model.logits(b2.to_tensor(copy["players"][sub])[None],
                          b2.to_tensor(copy["ball"][sub])[None],
                          b2.to_tensor(norm_ctx(copy["ctx"][sub], stats))[None],
                          b2.to_tensor(copy["dref"][sub])[None])[0].numpy()
        rep = np.minimum((idx - idx[0]) // SUB, len(sub) - 1)
        pa[idx] = 1 / (1 + np.exp(-lg[rep, 0]))
        ph[idx] = 1 / (1 + np.exp(-lg[rep, 1]))
    model.train()
    return pa, ph


def train_fold(i, steps, seed=0, cfg=None, refit_steps=None):
    """Train on fold i and predict its held-out sets. With `refit_steps` there is no validation
    split: all training matches of the fold are used for exactly that many steps (the final
    candidate, after a validated run fixed the number of steps)."""
    global EXTRA, LABELS, BALL
    cfg = {**DEFAULT, **(cfg or {})}
    EXTRA = bool(cfg["player_extra"])
    LABELS = cfg["labels"]
    BALL = bool(cfg["ball_head"])
    win = int(cfg["win_s"] * 10 / SUB)
    test_dfl, test_sc = b1_dev.folds()[i]
    train_dfl = [m for m in b1_dev.SPLIT["dfl"]["dev"] if m != test_dfl]
    train_sc = [m for m in b1_dev.SPLIT["skillcorner"]["dev"] if m not in test_sc]
    train_sc_all = list(train_sc)  # the ball head may use them unlabelled even without SkillCorner
    if not cfg["use_sc"]:
        train_sc = []
    if refit_steps is None:
        val_dfl, train_dfl = train_dfl[-1:], train_dfl[:-1]
        val_sc, train_sc = train_sc[-2:], train_sc[:-2]
    else:
        val_dfl, val_sc, steps = [], [], refit_steps
    dfl = [dfl_copy(m, s, w) for m in train_dfl for s in range(b1_dev.N_DEGRADE)
           for w in (False, True)]
    sc = [sc_copy(m, w) for m in train_sc for w in (False, True)]
    stats = ctx_stats(dfl + sc)
    vdfl = [dfl_copy(m, 0, False) for m in val_dfl]
    vsc = [sc_copy(m, False) for m in val_sc]
    wd = [(ci, s) for ci, c in enumerate(dfl) for s in windows(c, win)]
    ws = [(ci, s) for ci, c in enumerate(sc) for s in windows(c, win)]
    if BALL:
        # unlabelled copies of the training SkillCorner matches of the fold with extra ball gaps
        gaps = [gap_copy(m) for m in train_sc_all]
        wg = [(ci, s) for ci, c in enumerate(gaps) for s in windows(c, win)]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = b2.B2(N_CTX, d=cfg["d"], heads=cfg["heads"], hidden=cfg["hidden"],
                  social=cfg["social"], k_near=cfg["k_near"], temporal=cfg["temporal"],
                  change_head=cfg["change_head"], ball_head=BALL,
                  n_player_feats=tokens.N_PLAYER_FEATS + (tokens.N_EXTRA_PLAYER_FEATS if EXTRA else 0))
    # validation loss is always the plain loss, so early stopping compares like with like
    base = {**cfg, "boundary_w": 1.0, "change_head": False}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / WARMUP))
    bs = cfg["batch"]
    n_sc = int(round(bs * SC_SHARE)) if cfg["use_sc"] else 0
    pick = lambda ws_, n: [ws_[k] for k in np.linspace(0, len(ws_) - 1, n).astype(int)] if ws_ else []
    if refit_steps is None:
        vwd = pick([(ci, s) for ci, c in enumerate(vdfl) for s in windows(c, win)], VAL_WINDOWS)
        vws = pick([(ci, s) for ci, c in enumerate(vsc) for s in windows(c, win)], VAL_WINDOWS // 2)
    best, best_state, best_step, bad = np.inf, None, steps, 0
    recent = []
    for step in range(steps):
        pd_ = [wd[k] for k in rng.choice(len(wd), bs - n_sc, replace=False)]
        ps_ = [ws[k] for k in rng.choice(len(ws), n_sc, replace=False)]
        # each group's loss is a mean over its frames; SkillCorner gets half the weight
        if BALL:
            pl_, bl_, cx_, dr_, y_, bxy_, bm_ = batch(dfl, pd_, stats, win, with_ball=True)
            loss = loss_dfl(model, pl_, bl_, cx_, dr_, y_, aux=cfg["aux"], cfg=cfg, bxy=bxy_, bmask=bm_)
            pg_ = [wg[k] for k in rng.choice(len(wg), cfg["ball_windows"], replace=False)]
            g = batch(gaps, pg_, stats, win, with_ball=True)
            loss = loss + cfg["ball_w"] * loss_ball(model.logits(*g[:4]), g[5], g[6])
        else:
            loss = loss_dfl(model, *batch(dfl, pd_, stats, win), aux=cfg["aux"], cfg=cfg)
        if n_sc:
            loss = loss + b1_dev.SC_TOTAL_WEIGHT * loss_sc(model, *batch(sc, ps_, stats, win))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        recent.append(loss.item())
        if (step + 1) % 50 == 0:  # loss curve: mean of the last 50 steps
            print(f"fold {i} step {step + 1} loss {np.mean(recent[-50:]):.4f}", flush=True)
        if refit_steps is None and (step + 1) % VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vl = np.mean([loss_dfl(model, *batch(vdfl, vwd[k:k + 8], stats, win),
                                       aux=cfg["aux"], cfg=base).item()
                              for k in range(0, len(vwd), 8)])
                if vws:
                    vl += b1_dev.SC_TOTAL_WEIGHT * np.mean(
                        [loss_sc(model, *batch(vsc, vws[k:k + 8], stats, win)).item()
                         for k in range(0, len(vws), 8)])
            model.train()
            print(f"fold {i} step {step + 1} val {vl:.4f}", flush=True)
            if vl < best - 1e-4:
                best, bad, best_step = vl, 0, step + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    print(f"fold {i} early stop at step {step + 1}, best val {best:.4f}", flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    out = {"_best_step": best_step,
           "_model": {"state": model.state_dict(), "cfg": cfg, "stats": stats}}
    tests = [(f"dfl_sim_{test_dfl}", dfl_copy(test_dfl, b1_dev.TEST_SEED, False), "dfl", test_dfl),
             (f"dfl_clean_{test_dfl}", dfl_copy(test_dfl, 0, False, clean=True), "dfl", test_dfl)]
    tests += [(f"sc_{s}", sc_copy(s, False), "skillcorner", s) for s in test_sc]
    for name, copy, src, mid in tests:
        m = b1_dev.load(src, mid)
        rows = (b1_dev.dfl_rows(mid, b1_dev.TEST_SEED, False) if name.startswith("dfl_sim")
                else b1_dev.dfl_rows(mid, 0, False, clean=True) if name.startswith("dfl")
                else b1_dev.sc_rows(mid, False))
        pa, ph = predict(model, copy, stats)
        out[name] = dict(p_alive=pa, p_home=ph, period=rows[3], near=rows[4],
                         gt=dataset.codes(m.alive, m.team, m.period), n=m.n_frames)
        if BALL and name.startswith("dfl_sim"):
            out["_ball_err"] = ball_error(model, copy, stats)
            print(f"fold {i} ball reconstruction error (m) where hidden:", out["_ball_err"], flush=True)
    return out


@torch.no_grad()
def ball_error(model, copy, stats):
    """Median / 90th percentile distance (m) between the reconstructed and the clean ball on the
    rows where the degraded ball is hidden."""
    model.eval()
    errs = []
    period = copy["period"]
    for per in np.unique(period[period > 0]):
        sub = np.flatnonzero(period == per)[::SUB]
        lg = model.logits(b2.to_tensor(copy["players"][sub])[None], b2.to_tensor(copy["ball"][sub])[None],
                          b2.to_tensor(norm_ctx(copy["ctx"][sub], stats))[None],
                          b2.to_tensor(copy["dref"][sub])[None])[0].numpy()
        m = copy["bmask"][sub]
        d = (lg[m, -2:] - copy["bxy"][sub][m]) * np.array(tokens.SCALE_POS)
        errs.append(np.hypot(d[:, 0], d[:, 1]))
    model.train()
    e = np.concatenate(errs)
    return [round(float(np.median(e)), 2), round(float(np.percentile(e, 90)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000, help="maximum steps (early stopping)")
    ap.add_argument("--tag", default="b2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--config", default="{}", help="JSON overrides of DEFAULT")
    ap.add_argument("--refit-from", default=None,
                    help="tag of a validated run: refit each fold on all its training matches "
                         "for that run's best number of steps")
    a = ap.parse_args()
    b2.set_threads(THREADS)
    steps_path = lambda tag: os.path.join(b1_dev.CACHE, f"steps_{tag}.json")
    refit = json.load(open(steps_path(a.refit_from))) if a.refit_from else {}
    pred, best = {}, {}
    for i in map(int, a.folds.split(",")):
        out = train_fold(i, a.steps, cfg=json.loads(a.config), refit_steps=refit.get(str(i)))
        best[str(i)] = out.pop("_best_step")
        os.makedirs(MODELS, exist_ok=True)
        torch.save(out.pop("_model"), os.path.join(MODELS, f"{a.tag}_f{i}.pt"))
        err = out.pop("_ball_err", None)
        if err is not None:
            json.dump(err, open(os.path.join(b1_dev.CACHE, f"ball_err_{a.tag}_f{i}.json"), "w"))
        pred.update(out)
        print("fold", i, "done, steps", best[str(i)], flush=True)
    pickle.dump(pred, open(os.path.join(b1_dev.CACHE, f"pred_{a.tag}.pkl"), "wb"))
    json.dump(best, open(steps_path(a.tag), "w"))


if __name__ == "__main__":
    main()
