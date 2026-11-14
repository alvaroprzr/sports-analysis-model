"""F4, part 1: decoder-level variants of E7 from the saved development predictions of its members
(no training, no new inference), each scored with the decoder cost chosen excluding the
evaluated fold (scripts/xfold.py), on both readings of the DFL labels:

    arith     equal-weight mean of the members' probabilities (current E7)
    geo       normalised geometric mean: per class, mean over the M members of log(max(p, EPS)),
              exponentiated and normalised over the two classes (EPS fixed beforehand)
    temp      each member's two probabilities recalibrated by a temperature on the logit before
              the arithmetic mean; the temperatures used for a held-out set are fitted (negative
              log-likelihood) on the held-out sets of the OTHER folds only
    dmin1     arith, then stoppages shorter than 1 s inside live play are given to the previous
              label (applied to the decoded frames)

    python scripts/f4_decoder.py > experiments/f4_decoder.md
"""

import os
import pickle
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import xfold  # noqa: E402
from poss import dataset, decode, evaluate  # noqa: E402

MEMBERS = ["b2e0_5f", "b2e1_5f", "b2e2_5f", "b2e5_5f"]
EPS = 1e-4
DMIN_S = 1.0


def load(tag):
    return pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))


def save(tag, pred):
    pickle.dump(pred, open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "wb"))
    for suffix in ("_ctl",):
        p = os.path.join(b1_dev.CACHE, f"pred_{tag}{suffix}.pkl")
        if os.path.exists(p):
            os.remove(p)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def geo(ps):
    l1 = np.mean([np.log(np.maximum(p, EPS)) for p in ps], 0)
    l0 = np.mean([np.log(np.maximum(1 - p, EPS)) for p in ps], 0)
    return 1 / (1 + np.exp(l0 - l1))


def fit_temperature(z, y):
    nll = lambda t: np.mean(np.logaddexp(0, -np.where(y, 1, -1) * z / t))
    return minimize_scalar(nll, bounds=(0.25, 4.0), method="bounded").x


def targets(r, k):
    g = r["gt"][r["near"]]
    alive = (g < dataset.X) if k.startswith("dfl_sim") else np.zeros(len(g), bool)
    live = g < dataset.D if (k.startswith("dfl_sim") or k.startswith("sc_")) else np.zeros(len(g), bool)
    return alive, g != dataset.D, live, g == dataset.H


def temperatures(preds, keys_fit):
    out = []
    for pr in preds:
        za, ya, zh, yh = [], [], [], []
        for k in keys_fit:
            r = pr[k]
            a_m, a_y, h_m, h_y = targets(r, k)
            za.append(logit(r["p_alive"][a_m]))
            ya.append(a_y[a_m])
            zh.append(logit(r["p_home"][h_m]))
            yh.append(h_y[h_m])
        out.append((fit_temperature(np.concatenate(za), np.concatenate(ya)),
                    fit_temperature(np.concatenate(zh), np.concatenate(yh))))
    return out


def variant(name):
    preds = [load(t) for t in MEMBERS]
    keys = list(preds[0])
    out, temps = {}, {}
    if name == "temp":
        for i in range(b1_dev.N_FOLDS):
            temps[i] = temperatures(preds, [k for k in keys if xfold.fold_of(k) != i])
    for k in keys:
        r = dict(preds[0][k])
        pa = [p[k]["p_alive"] for p in preds]
        ph = [p[k]["p_home"] for p in preds]
        if name == "geo":
            r["p_alive"], r["p_home"] = geo(pa), geo(ph)
        elif name == "temp":
            ts = temps[xfold.fold_of(k)]
            sig = lambda z: 1 / (1 + np.exp(-z))
            r["p_alive"] = np.mean([sig(logit(p) / t[0]) for p, t in zip(pa, ts)], 0)
            r["p_home"] = np.mean([sig(logit(p) / t[1]) for p, t in zip(ph, ts)], 0)
        else:
            r["p_alive"], r["p_home"] = np.mean(pa, 0), np.mean(ph, 0)
        out[k] = r
    return out, temps


def dmin(codes, n):
    """Stoppages shorter than n frames between two live frames take the previous label."""
    c = codes.copy()
    t = np.flatnonzero(np.diff(np.r_[-1, (c == dataset.D).astype(int), -1]) != 0)
    for a, b in zip(t[:-1], t[1:]):
        if c[a] == dataset.D and b - a < n and a > 0 and b < len(c) and \
                c[a - 1] in (dataset.H, dataset.A) and c[b] in (dataset.H, dataset.A):
            c[a:b] = c[a - 1]
    return c


def grid(tag, post=None):
    pred = load(tag)
    g = {}
    for cst in xfold.C_GRID:
        s = {}
        for k, r in pred.items():
            c = decode.to_native(decode.decode(r["p_alive"], r["p_home"], r["period"],
                                               (cst, xfold.C_STOP, xfold.C_STOP)), r["near"], r["n"])
            if post is not None:
                c = post(c)
            s[k] = (evaluate.full_scores(r["gt"], c)["spellF1"] if k.startswith("dfl")
                    else evaluate.live_scores(r["gt"], c)["spellF1_live"])
        g[cst] = s
    return g


def main():
    rows = {}
    for name in ("arith", "geo", "temp", "dmin1"):
        base = "f4_arith" if name == "dmin1" else f"f4_{name}"
        if name != "dmin1":
            pred, temps = variant(name)
            save(base, pred)
            if temps:
                print(f"Temperatures (alive, team) per member, fitted without each fold: "
                      + "; ".join(f"fold {i}: " + ", ".join(f"({a:.2f}, {h:.2f})" for a, h in t)
                                  for i, t in temps.items()) + "\n")
        post = (lambda c: dmin(c, int(DMIN_S * 25))) if name == "dmin1" else None
        for reading, t in (("flag", base), ("control", xfold.with_control(base))):
            rows[(name, reading)] = xfold.scores(t, grid(t, post))
    print("| variant | reading | c_team per fold | DFL sim | SC live | combined |\n|---|---|---|---|---|---|")
    for (name, reading), (s, ch) in rows.items():
        d, sc, cmb = xfold.summary(s)
        print(f"| {name} | {reading} | {list(ch.values())} | {d:.3f} | {sc:.3f} | {cmb:.3f} |")
    print("\nPaired differences against arith (combined: diff, SE; DFL sim; SC live):\n")
    for name in ("geo", "temp", "dmin1"):
        cells = []
        for reading in ("flag", "control"):
            a, b = rows[(name, reading)][0], rows[("arith", reading)][0]
            diff, se = xfold.combined_paired(a, b)
            md = xfold.paired(a, b, "dfl_sim")[0]
            ms = xfold.paired(a, b, "sc_")[0]
            cells.append(f"{reading}: {diff:+.4f} ({se:.4f}), DFL {md:+.4f}, SC {ms:+.4f}")
        print(f"- {name}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
