"""Does the frozen E7 find stoppages when the ball is NOT hidden during them? (diagnostic, external
data only, no training, nothing chosen; the provided match is not read).

Hypothesis: P(alive) is learnt on simulated DFL only; the simulator's ball gaps are calibrated on
broadcast tracking, where the ball goes missing mostly during stoppages, so the model may read
"long gap = stoppage, visible ball = play".

The fold models of the four E7 members (held-out fold, teams-swapped inference averaged) with the
submitted decoder (configs/b2_ens_dr.json) are scored on the five DFL development matches in four
regimes that differ only in the ball gaps:
    sim      the held-out degradation of development (gaps from the calibrated hazards);
    clean    DFL as provided (no smoothing, no gaps);
    nogaps   the same degradation without its ball-gap step (smoothing, track cuts, snaps kept);
    shuffled the gaps of `sim` (same number, same lengths, same visible share) at positions
             independent of the game: inside each period the visible runs and the gap runs are
             permuted separately and interleaved again.
SkillCorner (out-of-fold predictions, real tracking): the same frame-level figures against its
approximate dead-ball label, split by ball visible / missing.

CRITERION, fixed before any result: the failure counts as reproduced if the frame-level recall of D
in `nogaps`, pooled over the five matches, is below 0.8.

    POSS_SIM=ext_dr python scripts/gap_structure_check.py cached   # sim, clean, SkillCorner (no inference)
    POSS_SIM=ext_dr python scripts/gap_structure_check.py new      # nogaps, shuffled (inference)
    POSS_SIM=ext_dr python scripts/gap_structure_check.py why      # why SkillCorner differs

`why` (added after the first two showed where the miss is): does P(alive) still rank SkillCorner's
frames (AUC, level), does the miss depend on how many players SkillCorner really detects, and how
does the ball move while it is dead in each source (a ball frozen by the provider during stoppages
is a cue automatic tracking does not have).
"""

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
from make_submission_b2ens import ROUND, load_model  # noqa: E402
from poss import b2, ballout, dataset, decode, evaluate, simulate  # noqa: E402
from poss.match import Match  # noqa: E402

pm = evaluate.pm
MEMBERS = ("b2e0d", "b2e1d", "b2e2d", "b2e5d")
CONF = json.load(open(os.path.join(ROOT, "configs", "b2_ens_dr.json")))
OOF = os.path.join(b1_dev.CACHE, "pred_f4d_swap.pkl")
NEW = os.path.join(b1_dev.CACHE, "pred_gapcheck.pkl")
BINS = (0, 5, 10, 20, np.inf)
FPS = 25
_MATCH = {}


def runs(mask):
    t = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(t[::2], t[1::2]))


def degraded(mid, with_ball):
    key = (mid, with_ball)
    if key not in _MATCH:
        parts = ("smooth", "players", "snaps") + (("ball",) if with_ball else ())
        _MATCH[key] = simulate.degrade(b1_dev.load("dfl", mid), b1_dev.calibration(),
                                       np.random.default_rng(b1_dev.TEST_SEED), parts=parts)
    return _MATCH[key]


def shuffled(mid):
    full, nog = degraded(mid, True), degraded(mid, False)
    assert np.array_equal(np.isfinite(full.home), np.isfinite(nog.home))  # same players
    vis = np.isfinite(full.ball[:, 0])
    ball = nog.ball.copy()
    rng = np.random.default_rng([b1_dev.TEST_SEED, 1])
    for per in np.unique(full.period[full.period > 0]):
        idx = np.flatnonzero(full.period == per)
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
        ball[idx[~new]] = np.nan
    return Match(nog.name, nog.source + "+shuf", nog.fps, nog.pitch, nog.home, nog.away, ball,
                 nog.period, nog.alive, nog.team)


def new_copy(mid, regime, swap):
    def make():
        m = degraded(mid, False) if regime == "nogaps" else shuffled(mid)
        return dataset.swap_teams(m) if swap else m
    key = f"dfl_{mid}_{regime}_{b1_dev.SIM}_{int(swap)}"
    return b2_dev.copy_tokens(key, make, lambda: b1_dev.cached_rows(key, make))


def predict_new():
    b2.set_threads(b2_dev.THREADS)
    ref = pickle.load(open(OOF, "rb"))
    out = {}
    for i, (mid, _) in enumerate(b1_dev.folds()):
        base = {k: v for k, v in ref[f"dfl_sim_{mid}"].items() if k not in ("p_alive", "p_home")}
        copies = {r: (new_copy(mid, r, False), new_copy(mid, r, True)) for r in ("nogaps", "shuffled")}
        _MATCH.clear()
        acc = {r: ([], []) for r in copies}
        for tag in MEMBERS:
            model, stats = load_model(os.path.join(b2_dev.MODELS, f"{tag}_f{i}_f{i}.pt"))
            for r, (c0, c1) in copies.items():
                assert np.array_equal(c0["period"], base["period"])
                pa, ph = b2_dev.predict(model, c0, stats)
                pa2, ph2 = b2_dev.predict(model, c1, stats)
                acc[r][0].append((pa + pa2) / 2)
                acc[r][1].append((ph + 1 - ph2) / 2)
        for r in copies:
            out[f"dfl_{r}_{mid}"] = dict(base, p_alive=np.mean(acc[r][0], 0),
                                         p_home=np.mean(acc[r][1], 0), ball=copies[r][0]["ball"])
        print("predicted", mid, flush=True)
        pickle.dump(out, open(NEW, "wb"))


def decode_native(r, ball):
    pa = ballout.apply(r["p_alive"], ball, **CONF["ball_out"])
    pa, ph = np.round(pa, ROUND), np.round(r["p_home"], ROUND)
    c10 = decode.decode(pa, ph, r["period"], tuple(CONF["costs"]))
    return decode.to_native(c10, r["near"], r["n"])


def counts(gt, pred, vis):
    """Frame counts for the D figures, in total and by ball visible / missing, and stoppages
    found (predicted D covers >= 50 %) by duration and by visibility of the ball during them."""
    ok = gt < dataset.X
    c = {}
    for name, sel in (("all", ok), ("vis", ok & vis), ("miss", ok & ~vis)):
        dead, pd_ = (gt == dataset.D) & sel, (pred == dataset.D) & sel
        c[name] = np.array([(dead & pd_).sum(), dead.sum(), pd_.sum(), (~dead & sel & pd_).sum(),
                            (~dead & sel).sum()], float)
    st = []
    for a, b in runs((gt == dataset.D) & ok):
        st.append(((b - a) / FPS, vis[a:b].mean() >= 0.5, (pred[a:b] == dataset.D).mean() >= 0.5))
    return c, st


def report(title, sets, live_only=False):
    """sets: [(name, record, ball tokens, ground truths {reading: codes})]"""
    print(f"\n## {title}\n")
    tot, stops, spell, ad, share = {}, [], {}, [], []
    for name, r, ball, gts in sets:
        pred = decode_native(r, ball)
        vis = decode.to_native((ball[:, ballout.VIS] > 0.5).astype(np.int8), r["near"], r["n"]) == 1
        gt = next(iter(gts.values()))
        c, st = counts(gt, pred, vis)
        for k, v in c.items():
            tot[k] = tot.get(k, 0) + v
        stops += st
        ad.append(pm.alive_dead_f1(pm.scored_confusion(pm.confusion(gt, pred))))
        for reading, g in gts.items():
            s = (evaluate.live_scores(g, pred)["spellF1_live"] if live_only
                 else evaluate.full_scores(g, pred)["spellF1"])
            spell.setdefault(reading, []).append(s)
        ok = gt < dataset.X
        share.append((vis[ok].mean(), (pred[ok] == dataset.D).mean()))
    print("| frames | D recall | D precision | D F1 | predicted D among live frames | dead frames | live frames |")
    print("|---|---|---|---|---|---|---|")
    for k, lab in (("all", "all"), ("vis", "ball visible"), ("miss", "ball missing")):
        tp, nd, npd, fp, nl = tot[k]
        rec, prec = tp / max(nd, 1), tp / max(npd, 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        print(f"| {lab} | {rec:.3f} | {prec:.3f} | {f1:.3f} | {fp / max(nl, 1):.3f} | {int(nd)} | {int(nl)} |")
    miss_dead = tot["miss"][1] / tot["all"][1]
    miss_live = tot["miss"][4] / tot["all"][4]
    print(f"\n- ball missing on {miss_dead:.3f} of the dead frames and {miss_live:.3f} of the live frames; "
          f"visible share {np.mean([s[0] for s in share]):.3f}; predicted D share "
          f"{np.mean([s[1] for s in share]):.3f} (true {tot['all'][1] / (tot['all'][1] + tot['all'][4]):.3f})")
    print(f"- alive/dead F1 (mean over matches): {np.mean(ad):.3f}")
    for reading, v in spell.items():
        print(f"- spellF1 ({reading}): {np.mean(v):.3f}  [{', '.join(f'{x:.3f}' for x in v)}]")
    st = np.array(stops, dtype=object)
    print("\n| stoppage duration | n | found | ball mostly visible: n, found | ball mostly missing: n, found |")
    print("|---|---|---|---|---|")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = np.array([lo <= d < hi for d in st[:, 0]])
        v = sel & st[:, 1].astype(bool)
        m = sel & ~st[:, 1].astype(bool)
        f = lambda s: f"{int(s.sum())}, {st[s, 2].astype(bool).mean():.2f}" if s.sum() else "0, -"
        hi_s = f"{hi:g}" if np.isfinite(hi) else "inf"
        print(f"| {lo:g}-{hi_s} s | {int(sel.sum())} | {st[sel, 2].astype(bool).mean():.2f} | {f(v)} | {f(m)} |")
    return tot["all"][0] / tot["all"][1]


def dfl_sets(pred, regime, ball_of):
    out = []
    for mid, _ in b1_dev.folds():
        r = pred[f"dfl_{regime}_{mid}"]
        out.append((mid, r, ball_of(mid, r), {"flag": r["gt"], "control": b2_dev.control_codes(mid)[0]}))
    return out


def auc(score, pos):
    order = np.argsort(score, kind="stable")
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return (rank[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def why():
    pred = pickle.load(open(OOF, "rb"))
    pct = lambda x: ", ".join(f"{v:.3f}" for v in np.percentile(x, [5, 25, 50, 75, 95]))
    print("\n## P(alive) of the frozen E7 before decoding (10 Hz rows, out of fold)\n")
    print("| source | AUC of dead vs live | P(alive) on dead rows, percentiles 5 / 25 / 50 / 75 / 95 | on live rows "
          "| recall of D at P(alive) < 0.5 | < 0.9 | < 0.99 | predicted D among live at < 0.5 | < 0.9 | < 0.99 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for src in ("dfl_sim", "dfl_clean", "sc_"):
        pa, dead = [], []
        for k, r in pred.items():
            if k.startswith(src):
                g = r["gt"][r["near"]]
                ok = (g < dataset.X) & (r["period"] > 0)
                pa.append(r["p_alive"][ok]), dead.append(g[ok] == dataset.D)
        pa, dead = np.concatenate(pa), np.concatenate(dead)
        rec = " | ".join(f"{(pa[dead] < t).mean():.3f}" for t in (0.5, 0.9, 0.99))
        fd = " | ".join(f"{(pa[~dead] < t).mean():.3f}" for t in (0.5, 0.9, 0.99))
        print(f"| {src.strip('_')} | {auc(-pa, dead):.3f} | {pct(pa[dead])} | {pct(pa[~dead])} | {rec} | {fd} |")

    print("\n## SkillCorner by the number of players really detected (the others are extrapolated)\n")
    edges, lab = [0, 1, 8, 14, 18, 23], ["0", "1-7", "8-13", "14-17", "18-22"]
    acc = np.zeros((5, 5))
    for k, r in pred.items():
        if not k.startswith("sc_"):
            continue
        m = b1_dev.load("skillcorner", k[3:])
        ndet = m.meta["detected_home"].sum(1) + m.meta["detected_away"].sum(1)
        nat = decode_native(r, b2_dev.sc_copy(k[3:], False)["ball"])
        g = r["gt"]
        bn = np.searchsorted(edges, ndet, side="right") - 1
        for b in range(5):
            sel = (g < dataset.X) & (bn == b)
            dead, pd_ = sel & (g == dataset.D), sel & (nat == dataset.D)
            acc[b] += [sel.sum(), dead.sum(), (dead & pd_).sum(), pd_.sum(), (pd_ & ~dead).sum()]
    print("| players detected | share of frames | dead share | D recall | D precision | predicted D among live |")
    print("|---|---|---|---|---|---|")
    for b in range(5):
        n, nd, tp, npd, fp = acc[b]
        print(f"| {lab[b]} | {n / acc[:, 0].sum():.3f} | {nd / n:.3f} | {tp / nd:.3f} | {tp / max(npd, 1):.3f} "
              f"| {fp / (n - nd):.3f} |")

    print("\n## The visible ball while dead and while live, by source (clean tracking as provided)\n")
    print("| source | state | frame-to-frame speed m/s, percentiles 10 / 25 / 50 / 75 / 90 | exactly 0 | < 0.5 m/s "
          "| a really observed player within 1.5 m |")
    print("|---|---|---|---|---|---|")
    for src in ("dfl", "skillcorner"):
        sp, near = {0: [], 1: []}, {0: [], 1: []}
        for mid in b1_dev.SPLIT[src]["dev"]:
            m = b1_dev.load(src, mid)
            B = m.ball[:, :2].astype(np.float64)
            step = np.linalg.norm(B[1:] - B[:-1], axis=1) * m.fps
            P = np.concatenate([m.home, m.away], 1).astype(np.float64)
            if src == "skillcorner":
                P[~np.concatenate([m.meta["detected_home"], m.meta["detected_away"]], 1)] = np.nan
            d = np.linalg.norm(P - B[:, None], axis=2)
            d = np.where(np.isfinite(d), d, np.inf).min(1)
            for a in (0, 1):
                sel = (m.alive[1:] == a) & (m.alive[:-1] == a) & (m.period[1:] > 0) & np.isfinite(step)
                sp[a].append(step[sel])
                near[a].append(d[(m.alive == a) & (m.period > 0) & np.isfinite(B[:, 0])])
        for a, name in ((0, "dead"), (1, "live")):
            v, d = np.concatenate(sp[a]), np.concatenate(near[a])
            q = ", ".join(f"{x:.2f}" for x in np.percentile(v, [10, 25, 50, 75, 90]))
            print(f"| {src} | {name} | {q} | {(v == 0).mean():.3f} | {(v < 0.5).mean():.3f} | {(d < 1.5).mean():.3f} |")


def main():
    assert b1_dev.SIM == "ext_dr"
    if sys.argv[1] == "cached":
        pred = pickle.load(open(OOF, "rb"))
        print("Frozen E7 (fold models, teams-swapped inference averaged, submitted decoder) on held-out "
              "development sets. Criterion fixed beforehand: reproduced if D recall in `nogaps` < 0.8.")
        report("DFL sim (held-out degradation of development)",
               dfl_sets(pred, "sim", lambda mid, r: b2_dev.dfl_copy(mid, b1_dev.TEST_SEED, False)["ball"]))
        report("DFL clean (as provided)",
               dfl_sets(pred, "clean", lambda mid, r: b2_dev.dfl_copy(mid, 0, False, clean=True)["ball"]))
        sc = [(k, r, b2_dev.sc_copy(k[3:], False)["ball"], {"live": r["gt"]})
              for k, r in pred.items() if k.startswith("sc_")]
        report("SkillCorner, 17 development matches (approximate dead-ball label; spellF1 live)", sc,
               live_only=True)
    elif sys.argv[1] == "why":
        why()
    else:
        if not os.path.exists(NEW):
            predict_new()
        pred = pickle.load(open(NEW, "rb"))
        rec = report("DFL nogaps (degradation without its ball-gap step)",
                     dfl_sets(pred, "nogaps", lambda mid, r: r["ball"]))
        report("DFL shuffled (the gaps of `sim` at positions independent of the game)",
               dfl_sets(pred, "shuffled", lambda mid, r: r["ball"]))
        print(f"\nCriterion: D recall in `nogaps` = {rec:.3f} -> "
              + ("REPRODUCED (< 0.8)" if rec < 0.8 else "not reproduced (>= 0.8)"))


if __name__ == "__main__":
    main()
