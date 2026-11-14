"""B': a stoppage detector that does not rest on the DFL ball being frozen while dead
(experiments/gap_structure_check.md). Only P(alive) is replaced; P(home | alive) stays E7's.
External data only. Design and adoption rule fixed on 13 November before any result
(scripts/alive_select.py applies the rule).

Training data:
  SkillCorner  real broadcast tracking with its approximate dead-ball label; only rows with at least
               MIN_DET really detected players, the players SkillCorner did not detect masked as
               missing before the features are computed (its extrapolation is not an observation,
               and 72 % of its dead frames have no detected player at all);
  DFL          with the ball hidden while it is dead, BEFORE the simulator degrades the match (the
               provider keeps the dead ball at fixed coordinates: not an observation); training
               copies with the domain-randomised simulator, held-out copy at the calibrated one.
Features: the B1 rows without the "tracked" columns (how many players are tracked depends on the
camera, not on the game). LightGBM with the parameters of B1. Predictions are the mean of the match
and of the match with the teams exchanged.

Arms (five folds, the folds of b1_dev):
  a  SkillCorner only        b  DFL only        c  both (SkillCorner with half the total weight)
(arm d, a recalibration of E7's own P(alive), needs no training: scripts/alive_select.py)
  g  c plus, for every DFL training copy, a copy with the ball gaps at positions independent of the
     game (added on 13 November after check T2 of scripts/alive_checks.py failed for c; one
     training per fold, rule in scripts/alive_select2.py fixed before training)

    POSS_SIM=ext_dr python scripts/alive_dev.py rows      # feature rows and tokens of the new copies
    POSS_SIM=ext_dr python scripts/alive_dev.py e7        # E7 fold models on the held-out DFL copies
    POSS_SIM=ext_dr python scripts/alive_dev.py train a   # out-of-fold P(alive) of an arm
    POSS_SIM=ext_dr python scripts/alive_dev.py final c OUT.txt   # the model on all development matches
    POSS_SIM=ext_dr python scripts/alive_dev.py stress    # arm c and E7 on DFL with many ball gaps in live
                                                          # play (check T2 of scripts/alive_checks.py)
    POSS_SIM=ext_dr python scripts/alive_dev.py rows_g    # the extra DFL training copies of arm g
    POSS_SIM=ext_dr python scripts/alive_dev.py train g   # also predicts the stress copies
"""

import os
import pickle
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
from poss import dataset, simulate  # noqa: E402
from poss.match import Match  # noqa: E402

# Params
MIN_DET = 14  # really detected players for a SkillCorner row to count (fixed 13/11)
ROW_STEP = 2  # training rows at 5 Hz, as B1
E7HID = os.path.join(b1_dev.CACHE, "pred_alive_e7hid.pkl")
_NDET, _MATCH = {}, {}


def pred_path(arm):
    return os.path.join(b1_dev.CACHE, f"pred_alive_{arm}.pkl")


def keep_cols(cols):
    return np.array([not str(c).startswith("tracked t") for c in cols])


# -- SkillCorner with the undetected players masked ---------------------------------------------

def sc_det_match(mid, swap):
    m = b1_dev.load("skillcorner", mid)
    home, away = m.home.copy(), m.away.copy()
    home[~m.meta["detected_home"]] = np.nan
    away[~m.meta["detected_away"]] = np.nan
    m = Match(m.name, m.source + "+det", m.fps, m.pitch, home, away, m.ball, m.period, m.alive, m.team)
    return dataset.swap_teams(m) if swap else m


def sc_det_rows(mid, swap):
    return b1_dev.cached_rows(f"scdet_{mid}_{int(swap)}", lambda: sc_det_match(mid, swap),
                              token_rows=True)


def n_detected(mid):
    """Really detected players per native frame of a SkillCorner match."""
    if mid not in _NDET:
        m = b1_dev.load("skillcorner", mid)
        _NDET[mid] = (m.meta["detected_home"].sum(1) + m.meta["detected_away"].sum(1)).astype(np.int16)
    return _NDET[mid]


# -- DFL with the dead ball hidden --------------------------------------------------------------

def dfl_hid_match(mid, seed, swap):
    key = (mid, seed)
    if key not in _MATCH:
        m = b1_dev.load("dfl", mid)
        ball = m.ball.copy()
        ball[m.alive == 0] = np.nan
        m = Match(m.name, m.source + "+hid", m.fps, m.pitch, m.home, m.away, ball, m.period,
                  m.alive, m.team)
        spread = 0.0 if seed == b1_dev.TEST_SEED else b1_dev.SPREAD
        _MATCH.clear()
        _MATCH[key] = simulate.degrade(m, b1_dev.copy_calibration(mid, seed),
                                       np.random.default_rng(seed), spread=spread)
    d = _MATCH[key]
    return dataset.swap_teams(d) if swap else d


def hid_key(mid, seed, swap):
    return f"dflhid_{mid}_sim{seed}_{b1_dev.SIM}_{int(swap)}"


def dfl_hid_rows(mid, seed, swap):
    return b1_dev.cached_rows(hid_key(mid, seed, swap), lambda: dfl_hid_match(mid, seed, swap))


def dfl_hid_tokens(mid, swap):
    import b2_dev
    s = b1_dev.TEST_SEED
    return b2_dev.copy_tokens(hid_key(mid, s, swap), lambda: dfl_hid_match(mid, s, swap),
                              lambda: dfl_hid_rows(mid, s, swap))


# -- DFL training copies with the ball gaps at positions independent of the game (arm g) --------

def permuted_runs(vis, period, rng):
    """The visible runs and the gap runs of each period permuted separately and interleaved again."""
    new_all = vis.copy()
    for per in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == per)
        v = vis[idx]
        cut = np.flatnonzero(np.diff(v.astype(np.int8))) + 1
        lens = np.diff(np.r_[0, cut, len(v)])
        kind = v[np.r_[0, cut]]
        seq = {True: list(rng.permutation(lens[kind])), False: list(rng.permutation(lens[~kind]))}
        new, k, cur = np.zeros(len(v), bool), 0, bool(kind[0])
        for _ in range(len(lens)):
            n = int(seq[cur].pop())
            new[k:k + n] = cur
            k += n
            cur = not cur
        assert k == len(v) and new.sum() == v.sum()
        new_all[idx] = new
    return new_all


def indep_match(mid, seed, swap):
    """The gaps of the usual copy of the match as provided (same simulator and seed), moved to
    positions independent of the game, on the copy of the match with the dead ball hidden."""
    key = ("indep", mid, seed)
    if key not in _MATCH:
        m = b1_dev.load("dfl", mid)
        cal = b1_dev.copy_calibration(mid, seed)
        full = simulate.degrade(m, cal, np.random.default_rng(seed), spread=b1_dev.SPREAD)
        vis = np.isfinite(full.ball[:, 0])
        del full
        ball = m.ball.copy()
        ball[m.alive == 0] = np.nan
        hid = Match(m.name, m.source + "+hid", m.fps, m.pitch, m.home, m.away, ball, m.period,
                    m.alive, m.team)
        d = simulate.degrade(hid, cal, np.random.default_rng(seed), spread=b1_dev.SPREAD,
                             parts=("smooth", "players", "snaps"))
        new = permuted_runs(vis, m.period, np.random.default_rng([seed, 1]))
        ball = d.ball.copy()
        ball[(m.period > 0) & ~new] = np.nan
        live = (m.period > 0) & (m.alive == 1)
        print(f"indep {mid} seed {seed}: live frames without the ball "
              f"{np.isnan(ball[live, 0]).mean():.3f}, visible share of the usual copy "
              f"{vis[m.period > 0].mean():.3f}", flush=True)
        _MATCH.clear()
        _MATCH[key] = Match(d.name, d.source + "+indep", d.fps, d.pitch, d.home, d.away, ball,
                            d.period, d.alive, d.team)
    d = _MATCH[key]
    return dataset.swap_teams(d) if swap else d


def indep_rows(mid, seed, swap):
    return b1_dev.cached_rows(f"dflindep_{mid}_sim{seed}_{b1_dev.SIM}_{int(swap)}",
                              lambda: indep_match(mid, seed, swap))


def build_rows_g():
    for mid in b1_dev.SPLIT["dfl"]["dev"]:
        for seed in range(b1_dev.N_DEGRADE):
            for swap in (False, True):
                indep_rows(mid, seed, swap)
            print("rows", mid, seed, flush=True)


def build_rows():
    for mid in b1_dev.SPLIT["skillcorner"]["dev"]:
        for swap in (False, True):
            sc_det_rows(mid, swap)
        print("rows", mid, flush=True)
    for mid in b1_dev.SPLIT["dfl"]["dev"]:
        for seed in list(range(b1_dev.N_DEGRADE)) + [b1_dev.TEST_SEED]:
            for swap in (False, True):
                dfl_hid_rows(mid, seed, swap)
                if seed == b1_dev.TEST_SEED:
                    dfl_hid_tokens(mid, swap)
            print("rows", mid, seed, flush=True)


# -- E7 on the held-out DFL copies (reference and team head) ------------------------------------

def e7_hid():
    import b2_dev
    from make_submission_b2ens import load_model
    from poss import b2
    members = ("b2e0d", "b2e1d", "b2e2d", "b2e5d")
    b2.set_threads(b2_dev.THREADS)
    out = {}
    for i, (mid, _) in enumerate(b1_dev.folds()):
        c0, c1 = dfl_hid_tokens(mid, False), dfl_hid_tokens(mid, True)
        _, _, _, period, near = dfl_hid_rows(mid, b1_dev.TEST_SEED, False)
        m = b1_dev.load("dfl", mid)
        pas, phs = [], []
        for tag in members:
            model, stats = load_model(os.path.join(b2_dev.MODELS, f"{tag}_f{i}_f{i}.pt"))
            pa, ph = b2_dev.predict(model, c0, stats)
            pa2, ph2 = b2_dev.predict(model, c1, stats)
            pas.append((pa + pa2) / 2), phs.append((ph + 1 - ph2) / 2)
        out[f"dfl_hid_{mid}"] = dict(p_alive=np.mean(pas, 0), p_home=np.mean(phs, 0), period=period,
                                     near=near, n=m.n_frames, ball=c0["ball"],
                                     gt=dataset.codes(m.alive, m.team, m.period))
        print("e7", mid, flush=True)
    pickle.dump(out, open(E7HID, "wb"))


# -- the alive classifier -----------------------------------------------------------------------

def training_set(arm, train_dfl, train_sc):
    pd_, ps_, kc = [], [], None
    if arm in ("b", "c", "g"):
        for make in (dfl_hid_rows, indep_rows)[:2 if arm == "g" else 1]:
            for mid in train_dfl:
                for seed in range(b1_dev.N_DEGRADE):
                    for swap in (False, True):
                        F, cols, y, _, _ = make(mid, seed, swap)
                        kc = keep_cols(cols) if kc is None else kc
                        sel = np.flatnonzero(y < dataset.X)[::ROW_STEP]
                        pd_.append((F[sel][:, kc], y[sel]))
                        del F
    if arm in ("a", "c", "g"):
        for mid in train_sc:
            for swap in (False, True):
                F, cols, y, _, near = sc_det_rows(mid, swap)
                kc = keep_cols(cols) if kc is None else kc
                sel = np.flatnonzero((y < dataset.X) & (n_detected(mid)[near] >= MIN_DET))[::ROW_STEP]
                ps_.append((F[sel][:, kc], y[sel]))
                del F
    nd, ns = sum(len(y) for _, y in pd_), sum(len(y) for _, y in ps_)
    X, y = b1_dev._stack(pd_ + ps_, int(kc.sum()))
    w = None
    if nd and ns:
        w = np.r_[np.ones(nd), np.full(ns, b1_dev.SC_TOTAL_WEIGHT * nd / ns)]
    return X, (y != dataset.D).astype(int), w, kc, (nd, ns)


def p_alive(clf, kc, rows_plain, rows_swap):
    return (clf.predict_proba(rows_plain[0][:, kc])[:, 1]
            + clf.predict_proba(rows_swap[0][:, kc])[:, 1]) / 2


def train(arm):
    out = {}
    for i, (test_dfl, test_sc) in enumerate(b1_dev.folds()):
        X, y, w, kc, n = training_set(arm, [m for m in b1_dev.SPLIT["dfl"]["dev"] if m != test_dfl],
                                      [m for m in b1_dev.SPLIT["skillcorner"]["dev"] if m not in test_sc])
        clf = lgb.LGBMClassifier(**b1_dev.GBM).fit(X, y, sample_weight=w)
        del X
        s = b1_dev.TEST_SEED
        out[f"dfl_hid_{test_dfl}"] = p_alive(clf, kc, dfl_hid_rows(test_dfl, s, False),
                                             dfl_hid_rows(test_dfl, s, True))
        for mid in test_sc:
            out[f"sc_{mid}"] = p_alive(clf, kc, sc_det_rows(mid, False), sc_det_rows(mid, True))
        if arm == "g":
            out[f"dfl_stress_{test_dfl}"] = p_alive(clf, kc, stress_rows(test_dfl, False),
                                                    stress_rows(test_dfl, True))
        print("fold", i, "rows DFL / SC", n, "dead share", round(1 - y.mean(), 3), flush=True)
    pickle.dump(out, open(pred_path(arm), "wb"))


def final(arm, path):
    X, y, w, kc, n = training_set(arm, b1_dev.SPLIT["dfl"]["dev"], b1_dev.SPLIT["skillcorner"]["dev"])
    clf = lgb.LGBMClassifier(**b1_dev.GBM).fit(X, y, sample_weight=w)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    clf.booster_.save_model(path)
    print("saved", path, "rows DFL / SC", n, "features", int(kc.sum()))


# -- stress regime: ball gaps at positions independent of the game, dead ball hidden -------------

STRESS = os.path.join(b1_dev.CACHE, "pred_alive_stress.pkl")


def stress_match(mid, swap):
    import gap_structure_check as g
    key = ("stress", mid)
    if key not in _MATCH:
        m = g.shuffled(mid)
        g._MATCH.clear()
        ball = m.ball.copy()
        ball[m.alive == 0] = np.nan
        _MATCH.clear()
        _MATCH[key] = Match(m.name, m.source + "+hid", m.fps, m.pitch, m.home, m.away, ball, m.period,
                            m.alive, m.team)
    d = _MATCH[key]
    return dataset.swap_teams(d) if swap else d


def stress_rows(mid, swap):
    return b1_dev.cached_rows(f"dflstress_{mid}_{b1_dev.SIM}_{int(swap)}", lambda: stress_match(mid, swap))


def stress_tokens(mid, swap):
    import b2_dev
    return b2_dev.copy_tokens(f"dflstress_{mid}_{b1_dev.SIM}_{int(swap)}", lambda: stress_match(mid, swap),
                              lambda: stress_rows(mid, swap))


def stress():
    import b2_dev
    from make_submission_b2ens import load_model
    from poss import b2
    b2.set_threads(b2_dev.THREADS)
    ref_c = pickle.load(open(pred_path("c"), "rb"))
    out = {}
    for i, (mid, test_sc) in enumerate(b1_dev.folds()):
        r0, r1 = stress_rows(mid, False), stress_rows(mid, True)
        c0, c1 = stress_tokens(mid, False), stress_tokens(mid, True)
        assert np.array_equal(c0["period"], r0[3])
        pas, phs = [], []
        for tag in ("b2e0d", "b2e1d", "b2e2d", "b2e5d"):
            model, stats = load_model(os.path.join(b2_dev.MODELS, f"{tag}_f{i}_f{i}.pt"))
            pa, ph = b2_dev.predict(model, c0, stats)
            pa2, ph2 = b2_dev.predict(model, c1, stats)
            pas.append((pa + pa2) / 2), phs.append((ph + 1 - ph2) / 2)
        X, y, w, kc, _ = training_set("c", [m for m in b1_dev.SPLIT["dfl"]["dev"] if m != mid],
                                      [m for m in b1_dev.SPLIT["skillcorner"]["dev"] if m not in test_sc])
        clf = lgb.LGBMClassifier(**b1_dev.GBM).fit(X, y, sample_weight=w)
        del X
        s = b1_dev.TEST_SEED
        same = p_alive(clf, kc, dfl_hid_rows(mid, s, False), dfl_hid_rows(mid, s, True))
        assert np.allclose(same, ref_c[f"dfl_hid_{mid}"], atol=1e-9), "not the fold model of the selection"
        m = b1_dev.load("dfl", mid)
        out[f"dfl_stress_{mid}"] = dict(period=r0[3], near=r0[4], n=m.n_frames, ball=c0["ball"],
                                        gt=dataset.codes(m.alive, m.team, m.period), ph=np.mean(phs, 0),
                                        pa={"frozen": np.mean(pas, 0), "c": p_alive(clf, kc, r0, r1)})
        print("stress", mid, flush=True)
        pickle.dump(out, open(STRESS, "wb"))


if __name__ == "__main__":
    assert b1_dev.SIM == "ext_dr"
    cmd = sys.argv[1]
    if cmd == "rows":
        build_rows()
    elif cmd == "rows_g":
        build_rows_g()
    elif cmd == "e7":
        e7_hid()
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "stress":
        stress()
    else:
        final(sys.argv[2], sys.argv[3])
