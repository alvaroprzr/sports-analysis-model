"""B1 development bench: LightGBM on physics features + three-state chain decoder.

Five folds; fold i holds out DFL dev match i and the SkillCorner dev matches j with j % 5 == i.
Two classifiers, so that an unreliable label can be masked out of its loss:
  P(alive)          DFL only (SkillCorner's dead ball is approximate);
  P(home | alive)   DFL + SkillCorner live frames, SkillCorner with half the total weight.
DFL training matches enter as several simulated degradations (seeds 0..N_DEGRADE-1, shared by
every fold that trains on the match) plus their team-swapped copies;
SkillCorner (real broadcast noise) enters as is, plus team swap. Held-out DFL is scored on one
simulated degradation (the target domain) and on the clean tracking; SkillCorner on its real data.

    python scripts/b1_dev.py train  ARM   # fit the folds, cache held-out probabilities
    python scripts/b1_dev.py decode ARM   # sweep the decoder costs on the cached probabilities

ARM says what the DFL training matches look like (the held-out data are the same for every arm,
so the arms are the downstream test of the simulator):
    clean     DFL as provided;
    filtered  DFL with the simulator's smoothing only (ball and player filters of the calibration);
    sim       DFL through the full simulator (several degradations, randomised intensity).
"""

import itertools
import json
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from poss import dataset, decode, evaluate, simulate  # noqa: E402
from poss.match import Match  # noqa: E402

# Params
N_FOLDS = 5
N_DEGRADE = 2  # simulated degradations per DFL training match
TEST_SEED = 999  # held-out DFL is scored on one degradation at the central (calibrated) intensity
SPREAD = 0.2  # domain randomisation of simulator rates and scales, exp(U(-0.2, 0.2))
RATE = 10.0  # working rate (Hz); `--rate 25` for the frequency experiment (DFL only)
TRAIN_ROWS_HZ = 5  # training rows per second at any working rate (neighbours are near duplicates)
ROW_STEP = 2  # set from RATE in main()
USE_SC = True  # `--no-sc` trains without SkillCorner (it is still scored at 10 Hz)
LEVELS = False  # `--sc-levels`: also score SkillCorner with extra ball gaps (SC_LEVELS)
SC_TOTAL_WEIGHT = 0.5  # SkillCorner total weight relative to DFL in the team model
GBM = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100,
           subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
           deterministic=True, num_threads=6, verbose=-1, random_state=0)
COSTS = {"c_team": [1, 2, 3, 4, 5, 6, 8, 12, 16, 24], "c_in": [2, 6, 12], "c_out": [2, 6, 12]}

CACHE = os.path.join(ROOT, "data", "cache", "b1")
SPLIT = json.load(open(os.path.join(ROOT, "configs", "splits.json")))


def load(src, mid):
    return Match.load(os.path.join(ROOT, "data", "external", f"{src}_v3", f"{mid}.npz"))


def cached_rows(key, make, token_rows=False):
    if RATE != 10:
        key += f"_r{RATE:g}"
    path = os.path.join(CACHE, "rows", key + ".npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return z["F"], list(z["cols"]), z["y"], z["period"], z["near"]
    m = make()
    # token_rows: as many rows as the B2 tokens (np.arange(0, n - 1, fps / rate)); at 10 fps this is
    # one row fewer than physics.to_rate gives, as the cached development rows of SkillCorner have
    n_rows = len(np.arange(0, m.n_frames - 1, m.fps / RATE)) if token_rows else None
    F, cols, y, period, near = dataset.rows(m, rate=RATE, n_rows=n_rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, F=F, cols=np.array(cols), y=y, period=period, near=near)
    return F, cols, y, period, near


# Which simulator degrades the DFL matches: the name of an externally calibrated one
# (scripts/sim_ext_calibrate.py, e.g. POSS_SIM=ext_dr). Cached rows and tokens carry the name.
SIM = os.environ.get("POSS_SIM", "")


def calibration():
    assert SIM, "set POSS_SIM to an externally calibrated simulator (scripts/sim_ext_calibrate.py)"
    name = f"sim_calibration_{SIM}.pkl"
    cal = pickle.load(open(os.path.join(ROOT, "data", "cache", name), "rb"))
    return cal["cal"] if isinstance(cal, dict) else cal


# Domain randomisation arm (POSS_SIM=ext_dr): each training copy gets its own intensity, the share
# of visible ball drawn uniformly in [DR_VIS] (clean to very degraded, no target figure) and the
# track-cut hazard multiplied by U(DR_TRACKS); held-out DFL copies use the calibrated intensity.
DR_VIS = (0.2, 1.0)
DR_TRACKS = (0.0, 3.0)


def copy_calibration(mid, seed):
    """Simulator of one DFL copy (the same for B1 rows and B2 tokens)."""
    cal = calibration()
    if SIM != "ext_dr" or seed == TEST_SEED:
        return cal
    from poss import sim_external
    table = pickle.load(open(os.path.join(ROOT, "data", "cache", "sim_calibration_ext_dr.pkl"), "rb"))["table"]
    return sim_external.dr_copy(cal, table, mid, seed, DR_VIS, DR_TRACKS)


def dfl_rows(mid, seed, swap, clean=False, filtered=False):
    tag = "clean" if clean else "filt" if filtered else f"sim{seed}"
    if SIM and not clean:
        tag += f"_{SIM}"
    def make():
        m = load("dfl", mid)
        if filtered:
            m = simulate.degrade(m, calibration(), np.random.default_rng(seed), parts=("smooth",))
        elif not clean:
            spread = 0.0 if seed == TEST_SEED else SPREAD
            m = simulate.degrade(m, copy_calibration(mid, seed), np.random.default_rng(seed),
                                 spread=spread)
        return dataset.swap_teams(m) if swap else m
    return cached_rows(f"dfl_{mid}_{tag}_{int(swap)}", make)


SC_LEVELS = (0.7, 0.4)  # extra-degraded SkillCorner evaluation: share of visible ball


def sc_rows_level(mid, level):
    """A SkillCorner match with ball gaps added until the visible share of play frames is at most
    `level` (gaps of mean 3 s at random positions, fixed seed per match); same labels."""
    def make():
        m = load("skillcorner", mid)
        rng = np.random.default_rng(int(mid) % (2 ** 31))
        play = m.period > 0
        ball = m.ball.copy()
        while np.isfinite(ball[play, 0]).mean() > level:
            vis = np.flatnonzero(np.isfinite(ball[:, 0]) & play)
            st = rng.choice(vis)
            ball[st:st + int(np.ceil(rng.exponential(3.0) * m.fps))] = np.nan
        m.ball = ball
        return m
    return cached_rows(f"sc_{mid}_vis{int(level * 100)}", make)


def sc_rows(mid, swap):
    def make():
        m = load("skillcorner", mid)
        return dataset.swap_teams(m) if swap else m
    return cached_rows(f"sc_{mid}_{int(swap)}", make, token_rows=True)


def folds():
    dfl, sc = SPLIT["dfl"]["dev"], SPLIT["skillcorner"]["dev"]
    return [(dfl[i], [s for j, s in enumerate(sc) if j % N_FOLDS == i]) for i in range(N_FOLDS)]


def _stack(pieces, n_cols):
    """Concatenate (F, y) pieces into one preallocated array (no intermediate copies)."""
    n = sum(len(y) for _, y in pieces)
    X = np.empty((n, n_cols), np.float32)
    k = 0
    for F, y in pieces:
        X[k:k + len(y)] = F
        k += len(y)
    return X, np.concatenate([y for _, y in pieces]) if pieces else np.zeros(0, np.int8)


def train_fold(i, arm, params=GBM):
    """Fit P(alive) on DFL and P(home | alive) on DFL + SkillCorner live rows for fold i.

    Rows are filtered while loading and copied once into preallocated arrays: the naive version
    peaked at 5.6 GB, too close to the 7 GB available."""
    test_dfl, test_sc = folds()[i]
    train_dfl = [m for m in SPLIT["dfl"]["dev"] if m != test_dfl]
    train_sc = [m for m in SPLIT["skillcorner"]["dev"] if m not in test_sc] if USE_SC else []
    test_sc = test_sc if RATE == 10 else []  # SkillCorner is native 10 Hz: scored only there
    pieces = []
    for mid in train_dfl:
        for seed in range(N_DEGRADE if arm == "sim" else 1):
            for swap in (False, True):
                F, cols, y, _, _ = dfl_rows(mid, seed, swap, clean=arm == "clean",
                                            filtered=arm == "filtered")  # shared by the folds
                F, y = F[::ROW_STEP], y[::ROW_STEP]
                keep = y < dataset.X
                pieces.append((F[keep], y[keep]))
                del F
    n_cols = pieces[0][0].shape[1]
    Xd, yd = _stack(pieces, n_cols)
    del pieces
    alive = lgb.LGBMClassifier(**params).fit(Xd, (yd != dataset.D).astype(int))
    ld = yd < dataset.D
    pieces = [(Xd[ld], yd[ld])]
    del Xd
    nd = int(ld.sum())
    for mid in train_sc:
        for swap in (False, True):
            F, cols, y, _, _ = sc_rows(mid, swap)
            F, y = F[::ROW_STEP], y[::ROW_STEP]
            live = y < dataset.D
            pieces.append((F[live], y[live]))
            del F
    Xt, yt = _stack(pieces, n_cols)
    del pieces
    ns = len(yt) - nd
    wt = np.r_[np.ones(nd), np.full(ns, SC_TOTAL_WEIGHT * nd / max(ns, 1))]
    team = lgb.LGBMClassifier(**params).fit(Xt, (yt == dataset.H).astype(int), sample_weight=wt)
    del Xt
    out = {}
    for name, (F, _, y, period, near), m in (
            [(f"dfl_sim_{test_dfl}", dfl_rows(test_dfl, TEST_SEED, False), load("dfl", test_dfl)),
             (f"dfl_clean_{test_dfl}", dfl_rows(test_dfl, 0, False, clean=True),
              load("dfl", test_dfl))] +
            [(f"sc_{s}", sc_rows(s, False), load("skillcorner", s)) for s in test_sc] +
            [(f"scv{int(lv * 100)}_{s}", sc_rows_level(s, lv), load("skillcorner", s))
             for s in (test_sc if LEVELS else []) for lv in SC_LEVELS]):
        out[name] = dict(p_alive=alive.predict_proba(F)[:, 1], p_home=team.predict_proba(F)[:, 1],
                         period=period, near=near,
                         gt=dataset.codes(m.alive, m.team, m.period), n=m.n_frames)
    return out


def score(pred, costs):
    res = {"dfl_sim": [], "dfl_clean": [], "sc_live": [], "sc_full": [], "n_dfl": [], "gt_dfl": []}
    for name, r in pred.items():
        c10 = decode.decode(r["p_alive"], r["p_home"], r["period"], costs)
        c = decode.to_native(c10, r["near"], r["n"])
        if name.startswith("dfl"):
            s = evaluate.full_scores(r["gt"], c)
            key = "dfl_sim" if "_sim_" in name else "dfl_clean"
            res[key].append(s["spellF1"])
            if key == "dfl_sim":
                res["n_dfl"].append(s["n_pred"])
                res["gt_dfl"].append(s["n_gt"])
        else:
            res["sc_live"].append(evaluate.live_scores(r["gt"], c)["spellF1_live"])
            res["sc_full"].append(evaluate.full_scores(r["gt"], c)["spellF1"])
    return res


def main():
    global RATE, ROW_STEP, USE_SC
    os.makedirs(CACHE, exist_ok=True)
    arm = sys.argv[2]
    if "--rate" in sys.argv:
        RATE = float(sys.argv[sys.argv.index("--rate") + 1])
    USE_SC = "--no-sc" not in sys.argv
    global LEVELS
    LEVELS = "--sc-levels" in sys.argv
    ROW_STEP = int(round(RATE / TRAIN_ROWS_HZ))
    folds = ([int(x) for x in sys.argv[sys.argv.index("--folds") + 1].split(",")]
             if "--folds" in sys.argv else list(range(N_FOLDS)))
    tag = arm + (f"_r{RATE:g}" if RATE != 10 else "") + ("" if USE_SC else "_nosc") + \
        (f"_{SIM}" if SIM else "") + ("" if len(folds) == N_FOLDS else "_f" + "".join(map(str, folds))) + \
        ("_lv" if LEVELS else "")
    path = os.path.join(CACHE, f"pred_{tag}.pkl")
    if sys.argv[1] == "train":
        pred = {}
        for i in folds:
            pred.update(train_fold(i, arm))
            print("fold", i, "done", flush=True)
        pickle.dump(pred, open(path, "wb"))
        return
    pred = pickle.load(open(path, "rb"))
    print("| c_team | c_in | c_out | DFL sim spellF1 | DFL clean | SC live | SC full (approx D) "
          "| combined | DFL poss pred/gt |")
    print("|---|---|---|---|---|---|---|---|---|")
    for costs in itertools.product(*COSTS.values()):
        r = score(pred, costs)
        mean_se = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.3f}"
        print(f"| {costs[0]} | {costs[1]} | {costs[2]} | {mean_se(r['dfl_sim'])} | "
              f"{np.mean(r['dfl_clean']):.3f} | {mean_se(r['sc_live'])} | "
              f"{np.mean(r['sc_full']):.3f} | "
              f"{evaluate.combined(np.mean(r['dfl_sim']), np.mean(r['sc_live'])):.3f} | "
              f"{np.sum(r['n_dfl'])}/{np.sum(r['gt_dfl'])} |", flush=True)


if __name__ == "__main__":
    main()
