"""B': which P(alive), if any, replaces E7's (scripts/alive_dev.py). External data only. Rule fixed
on 13 November, before any of these predictions existed:

Candidates: P(alive) of an arm + P(home | alive) of E7 (out of fold) + the ball-out rule + a decoder
(c_team, c_stop) chosen by the zone score over three sources (DFL with the dead ball hidden, flag
and control readings; SkillCorner full) EXCLUDING the evaluated fold.
  a  LightGBM, SkillCorner only      b  LightGBM, DFL only      c  LightGBM, both
  d  E7's own P(alive) recalibrated: sigmoid(t * logit(p) + b), t and b fitted by logistic regression
     on the SkillCorner matches of the other folds (rows with >= 14 really detected players).
Reference: the frozen E7 with its submitted decoder on the same sets.

Adopt an arm only if
  1. SkillCorner full spellF1 (against its approximate dead-ball label) is better by more than 2
     paired SE (17 matches);
  2. SkillCorner live spellF1 is not worse by more than 1 paired SE;
  3. DFL with the dead ball hidden is not worse by more than 1 paired SE, flag and control readings.
Among the arms that pass: the best in 1; if d passes and is within 1 paired SE of the best, d (the
simplest). Neither the reserved test nor any frame of the provided match is read.

Metric 1, stated before any result: SkillCorner frames with fewer than 14 detected players are
neutralised, the same for every arm and for the reference (where the reference label says D the
prediction is set to D; where it says play and the prediction says D, the prediction abstains).
There the camera is away from the pitch (36.5 % of the frames have no detected player), no model
has training support and the provided match has nothing alike. The figure without neutralising is
reported as secondary.

    POSS_SIM=ext_dr python scripts/alive_select.py > experiments/alive_select.md
The last line is the decision as JSON.
"""

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import alive_dev  # noqa: E402
import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
import xfold  # noqa: E402
from poss import ballout, dataset, decode, decode2, evaluate  # noqa: E402

CONF = json.load(open(os.path.join(ROOT, "configs", "b2_ens_dr.json")))
OOF = os.path.join(b1_dev.CACHE, "pred_f4d_swap.pkl")
GRID = [(c, cs) for c in xfold.C_GRID for cs in (6.0, 2.0, 4.0, 8.0, 12.0)]
ARMS = ("a", "b", "c", "d")
NAMES = {"frozen": "E7 frozen", "a": "a: SkillCorner only", "b": "b: DFL only", "c": "c: both",
         "d": "d: E7 recalibrated"}
D, X = dataset.D, dataset.X
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_affine(z, y):
    """Logistic regression of y on z with an intercept (damped Newton steps, no regularisation)."""
    A = np.column_stack([z, np.ones(len(z))])

    def nll(w):
        m = A @ w
        return np.mean(np.logaddexp(0, m) - y * m)

    w = np.array([0.0, logit(np.mean(y))])
    for _ in range(200):
        p = 1 / (1 + np.exp(-A @ w))
        H = A.T @ (A * (p * (1 - p))[:, None]) / len(y) + 1e-12 * np.eye(2)
        step = np.linalg.solve(H, A.T @ (p - y) / len(y))
        t, f0 = 1.0, nll(w)
        while nll(w - t * step) > f0 and t > 1e-8:
            t /= 2
        w = w - t * step
        if np.abs(t * step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def auc(score, pos):
    order = np.argsort(score, kind="stable")
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return (rank[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_sets():
    e7 = pickle.load(open(alive_dev.E7HID, "rb"))
    oof = pickle.load(open(OOF, "rb"))
    sets = {}
    for mid, _ in b1_dev.folds():
        r = e7[f"dfl_hid_{mid}"]
        sets[f"dfl_hid_{mid}"] = dict(period=r["period"], near=r["near"], n=r["n"], gt=r["gt"],
                                      gt_ctl=b2_dev.control_codes(mid)[0], ball=r["ball"],
                                      ph=r["p_home"], pa={"frozen": r["p_alive"]})
    for k, r in oof.items():
        if k.startswith("sc_"):
            nd = alive_dev.n_detected(k[3:])
            assert len(nd) == r["n"]
            sets[k] = dict(period=r["period"], near=r["near"], n=r["n"], gt=r["gt"],
                           ball=b2_dev.sc_copy(k[3:], False)["ball"], ph=r["p_home"],
                           pa={"frozen": r["p_alive"]}, low=nd < alive_dev.MIN_DET)
    for arm in ("a", "b", "c"):
        p = pickle.load(open(alive_dev.pred_path(arm), "rb"))
        for k, s in sets.items():
            assert len(p[k]) == len(s["period"]), k
            s["pa"][arm] = p[k]
    for s in sets.values():
        y = s["gt"][s["near"]]
        s["ok10"] = (y < X) & (s["period"] > 0)
        if "low" in s:
            s["ok10"] &= ~s["low"][s["near"]]
        s["dead10"] = y == D
    params = {}
    for i in list(range(b1_dev.N_FOLDS)) + ["all"]:
        tr = [s for k, s in sets.items() if k.startswith("sc_") and (i == "all" or xfold.fold_of(k) != i)]
        z = np.concatenate([logit(s["pa"]["frozen"][s["ok10"]]) for s in tr])
        y = np.concatenate([(~s["dead10"][s["ok10"]]).astype(float) for s in tr])
        params[i] = fit_affine(z, y)
        if i != "all":
            t, b = params[i]
            for k, s in sets.items():
                if xfold.fold_of(k) == i:
                    s["pa"]["d"] = 1 / (1 + np.exp(-(t * logit(s["pa"]["frozen"]) + b)))
    return sets, params


_PREP = {}


def native(k, s, arm, c, cs, exact=False):
    if (k, arm) not in _PREP:
        pa = ballout.apply(s["pa"][arm], s["ball"], **CONF["ball_out"])
        _PREP[k, arm] = (np.round(pa, 4), np.round(s["ph"], 4))
    pa, ph = _PREP[k, arm]
    c10 = (decode.decode(pa, ph, s["period"], (c, cs, cs)) if exact
           else decode2.decode(pa, ph, s["period"], (c, cs, cs), k=1))
    return decode.to_native(c10, s["near"], s["n"])


def neutral(s, pred):
    p = pred.copy()
    p[s["low"] & (s["gt"] == D)] = D
    p[s["low"] & (s["gt"] < D) & (pred == D)] = X
    return p


def main_scores(k, s, pred):
    """The scores the decoder is chosen with: (flag, control) on DFL, metric 1 twice on SkillCorner."""
    if k.startswith("dfl"):
        return (evaluate.full_scores(s["gt"], pred)["spellF1"],
                evaluate.full_scores(s["gt_ctl"], pred)["spellF1"])
    m1 = evaluate.full_scores(s["gt"], neutral(s, pred))["spellF1"]
    return m1, m1


def frame_counts(s, pred, sel):
    dead, pd_ = sel & (s["gt"] == D), sel & (pred == D)
    return np.array([(dead & pd_).sum(), dead.sum(), pd_.sum(), (pd_ & ~dead).sum(), (sel & ~dead).sum()], float)


def zone(g, keys):
    rows = []
    for dec, sc in g.items():
        rows.append((dec, np.mean([sc[k][0] for k in keys if k.startswith("dfl")]),
                     np.mean([sc[k][1] for k in keys if k.startswith("dfl")]),
                     np.mean([sc[k][0] for k in keys if k.startswith("sc_")])))
    best = [max(r[i] for r in rows) for i in (1, 2, 3)]
    gaps = [max(best[0] - a, best[1] - b, best[2] - c) for _, a, b, c in rows]
    return rows[int(np.argmin(gaps))][0]


def evaluate_arm(sets, arm):
    """Per set, at the decoder chosen without its fold: every score and the frame counts."""
    keys = list(sets)
    if arm == "frozen":
        chosen = {i: tuple(CONF["costs"][:2]) for i in range(b1_dev.N_FOLDS)}
        final = chosen[0]
    else:
        g = {dec: {k: main_scores(k, sets[k], native(k, sets[k], arm, *dec)) for k in keys} for dec in GRID}
        chosen = {i: zone(g, [k for k in keys if xfold.fold_of(k) != i]) for i in range(b1_dev.N_FOLDS)}
        final = zone(g, keys)
    out = {}
    for k in keys:
        s = sets[k]
        pred = native(k, s, arm, *chosen[xfold.fold_of(k)], exact=True)
        ok = s["gt"] < X
        if k.startswith("dfl"):
            f, c = main_scores(k, s, pred)
            out[k] = dict(flag=f, control=c, frames=frame_counts(s, pred, ok))
        else:
            out[k] = dict(m1=main_scores(k, s, pred)[0],
                          live=evaluate.live_scores(s["gt"], pred)["spellF1_live"],
                          raw=evaluate.full_scores(s["gt"], pred)["spellF1"],
                          frames=frame_counts(s, pred, ok & ~s["low"]),
                          frames_all=frame_counts(s, pred, ok))
    return out, chosen, final


def paired(a, b, keys, field):
    d = np.array([a[k][field] - b[k][field] for k in keys])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main():
    assert b1_dev.SIM == "ext_dr"
    sets, params = load_sets()
    dk = [k for k in sets if k.startswith("dfl")]
    sk = [k for k in sets if k.startswith("sc_")]
    res = {arm: evaluate_arm(sets, arm) for arm in ("frozen",) + ARMS}
    ref = res["frozen"][0]
    mean = lambda r, keys, f: np.mean([r[k][f] for k in keys])

    print("B': replacing P(alive) only (scripts/alive_dev.py, scripts/alive_select.py). Five folds, held-out "
          "sets: 5 DFL (dead ball hidden, calibrated degradation) and 17 SkillCorner. Decoder chosen "
          "excluding the evaluated fold.\n")
    print("## Scores (spellF1)\n")
    print("| P(alive) | decoder per fold (c_team, c_stop) | SC full, metric 1 | SC live | DFL hidden, flag "
          "| DFL hidden, control | SC full without neutralising |")
    print("|---|---|---|---|---|---|---|")
    for arm, (r, chosen, _) in res.items():
        dec = ", ".join(f"({c}, {cs:g})" for c, cs in chosen.values())
        print(f"| {NAMES[arm]} | {dec} | {mean(r, sk, 'm1'):.4f} | {mean(r, sk, 'live'):.4f} | "
              f"{mean(r, dk, 'flag'):.4f} | {mean(r, dk, 'control'):.4f} | {mean(r, sk, 'raw'):.4f} |")

    print("\n## The rule: arm minus the frozen E7, paired by match (SE)\n")
    print("| P(alive) | 1. SC full, metric 1 (> 2 SE) | 2. SC live (>= -1 SE) | 3. DFL flag (>= -1 SE) "
          "| 3. DFL control (>= -1 SE) | passes |")
    print("|---|---|---|---|---|---|")
    passing = []
    for arm in ARMS:
        r = res[arm][0]
        d1, d2 = paired(r, ref, sk, "m1"), paired(r, ref, sk, "live")
        d3, d4 = paired(r, ref, dk, "flag"), paired(r, ref, dk, "control")
        ok = d1[0] > 2 * d1[1] and d2[0] >= -d2[1] and d3[0] >= -d3[1] and d4[0] >= -d4[1]
        if ok:
            passing.append(arm)
        f = lambda d: f"{d[0]:+.4f} ({d[1]:.4f})"
        print(f"| {NAMES[arm]} | {f(d1)} | {f(d2)} | {f(d3)} | {f(d4)} | {'YES' if ok else 'no'} |")

    winner = None
    if passing:
        winner = max(passing, key=lambda a: mean(res[a][0], sk, "m1"))
        if "d" in passing and winner != "d":
            dm, se = paired(res[winner][0], res["d"][0], sk, "m1")
            print(f"\nBest arm {winner} minus d on metric 1: {dm:+.4f} (SE {se:.4f})"
                  + (" -> within 1 SE, d (the simplest) is taken" if dm <= se else " -> more than 1 SE"))
            if dm <= se:
                winner = "d"
    print("\n**Decision by the rule: " + (f"arm {winner} ({NAMES[winner]}) ADOPTED**" if winner
                                         else "no arm adopted; the frozen E7 stays**"))

    print("\n## Stoppages, frame level, after decoding\n")
    print("| P(alive) | SC, >= 14 detected: D recall | D precision | predicted D among live | SC, all frames: "
          "D recall | D precision | DFL hidden: D recall | D precision | predicted D among live |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, (r, _, _) in res.items():
        a = sum(r[k]["frames"] for k in sk)
        b = sum(r[k]["frames_all"] for k in sk)
        c = sum(r[k]["frames"] for k in dk)
        print(f"| {NAMES[arm]} | {a[0] / a[1]:.3f} | {a[0] / max(a[2], 1):.3f} | {a[3] / a[4]:.3f} | "
              f"{b[0] / b[1]:.3f} | {b[0] / max(b[2], 1):.3f} | {c[0] / c[1]:.3f} | {c[0] / max(c[2], 1):.3f} "
              f"| {c[3] / c[4]:.3f} |")

    print("\n## P(alive) before decoding, 10 Hz rows (SkillCorner: rows with >= 14 detected players)\n")
    print("Cross-provider transfer: arm a never saw a DFL match, arm b never saw a SkillCorner match.\n")
    print("| P(alive) | SC: AUC | D recall at 0.5 | predicted D among live at 0.5 | DFL hidden: AUC "
          "| D recall at 0.5 | predicted D among live at 0.5 |")
    print("|---|---|---|---|---|---|---|")
    for arm in ("frozen",) + ARMS:
        cells = []
        for keys in (sk, dk):
            p = np.concatenate([sets[k]["pa"][arm][sets[k]["ok10"]] for k in keys])
            dead = np.concatenate([sets[k]["dead10"][sets[k]["ok10"]] for k in keys])
            cells += [f"{auc(-p, dead):.3f}", f"{(p[dead] < 0.5).mean():.3f}", f"{(p[~dead] < 0.5).mean():.3f}"]
        print(f"| {NAMES[arm]} | " + " | ".join(cells) + " |")

    print("\n## Arm d: recalibration fitted without the evaluated fold (temperature, bias)\n")
    print(", ".join(f"fold {i}: ({params[i][0]:.3f}, {params[i][1]:.3f})" for i in range(b1_dev.N_FOLDS))
          + f"; all 17 matches: ({params['all'][0]:.4f}, {params['all'][1]:.4f})")

    out = {"adopted": winner}
    if winner:
        c, cs = res[winner][2]
        out.update(costs=[c, cs, cs])
        if winner == "d":
            out.update(temperature=round(params["all"][0], 4), bias=round(params["all"][1], 4))
    print("\nDecoder on all folds per arm: "
          + ", ".join(f"{a}: ({res[a][2][0]}, {res[a][2][1]:g})" for a in ARMS))
    print("\n" + json.dumps(out))


if __name__ == "__main__":
    main()
