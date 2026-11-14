"""Quality of E7's alive / dead decision, separately from the team (diagnostic, no tuning).

DFL simulated, control reading (development held-out sets, E7 at the cost chosen excluding the
evaluated fold, adopted F4 decoder not applied):
  - frame-level F1, recall and precision of D on scored frames;
  - signed error of the start and end of each true stoppage (nearest predicted boundary of the
    same kind within +-3 s; predicted - true, s), and the share without a match;
  - recall of stoppages (a true stoppage counts as found if predicted D covers >= 50 % of it) by
    duration and by type, the type being the restart event that ends it in the DFL event file
    (throw-in, corner, goal kick, free kick, kick-off; others: no restart event within 5 s).
BAR-ATH, without labels (frames 11185-18684 excluded):
  - share of graded frames predicted D;
  - Smart Tagging set pieces whose interval, extended 5 s back, overlaps predicted D, by type;
  - episodes of the visible ball outside the pitch lines (> 0.5 m out for >= 0.5 s) with no
    predicted D within 3 s.

    python scripts/dead_ball_check.py CSV > experiments/dead_ball_check.md
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import b1_dev  # noqa: E402
import load_smart_tagging  # noqa: E402
import xfold  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from poss import barath, control, dataset, decode  # noqa: E402

FPS = 25
TOL = 3 * FPS
PUBLIC = (11185, 18684)
BINS = (0, 5, 10, 20, 40, np.inf)
RESTARTS = {"ThrowIn": "throw-in", "CornerKick": "corner", "GoalKick": "goal kick",
            "FreeKick": "free kick", "KickOff": "kick-off", "Penalty": "penalty"}


def runs(mask):
    t = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(t[::2], t[1::2]))


def nearest(true_f, pred_f):
    if not len(pred_f):
        return np.full(len(true_f), np.nan)
    j = np.clip(np.searchsorted(pred_f, true_f), 1, len(pred_f) - 1)
    cand = np.stack([pred_f[j - 1], pred_f[j]], 1)
    k = np.argmin(np.abs(cand - true_f[:, None]), 1)
    best = cand[np.arange(len(true_f)), k]
    err = (best - true_f).astype(float)
    err[np.abs(err) > TOL] = np.nan
    return err / FPS


def restart_frames(mid):
    """Tracking frame and type of each restart event of a DFL match (event clock aligned)."""
    m = b1_dev.load("dfl", mid)
    ev, ko = control.load_events(mid)
    sh = control.fit_shift(m, ev, ko)
    clock = control.half_clock(m)
    import xml.etree.ElementTree as ET
    path = control._files(mid)[0]
    out = []
    for e in ET.parse(path).getroot().findall("Event"):
        for c in e:
            if c.tag in RESTARTS:
                t = control._time(e.get("EventTime"))
                h = 2 if 2 in ko and t >= ko[2] else 1
                if h not in sh:
                    continue
                tt = t - ko[h] + sh[h][0]
                sel = np.flatnonzero(m.period == h)
                k = np.searchsorted(clock[sel], tt)
                if 0 <= k < len(sel):
                    out.append((int(sel[k]), RESTARTS[c.tag]))
    return out


def main():
    tag = xfold.with_control("ens_e7")
    _, chosen = xfold.scores(tag)
    pred = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))
    tp = fp = fn = 0
    err_s, err_e, found = [], [], []
    for k, r in pred.items():
        if not k.startswith("dfl_sim"):
            continue
        c = chosen[xfold.fold_of(k)]
        p = decode.to_native(decode.decode(r["p_alive"], r["p_home"], r["period"],
                                           (c, xfold.C_STOP, xfold.C_STOP)), r["near"], r["n"])
        g = r["gt"]
        ok = g < dataset.X
        tp += int(((g == dataset.D) & (p == dataset.D) & ok).sum())
        fp += int(((g != dataset.D) & (p == dataset.D) & ok).sum())
        fn += int(((g == dataset.D) & (p != dataset.D) & ok).sum())
        pd_ = runs((p == dataset.D) & ok)
        ps = np.array([a for a, _ in pd_])
        pe = np.array([b for _, b in pd_])
        rs = restart_frames(k.split("_")[-1])
        rf = np.array([f for f, _ in rs])
        for a, b in runs((g == dataset.D) & ok):
            if a == 0 or b >= len(g) or g[a - 1] == dataset.X or g[b] == dataset.X:
                continue
            cov = (p[a:b] == dataset.D).mean()
            near = np.flatnonzero(np.abs(rf - b) <= 5 * FPS) if len(rf) else np.array([], int)
            typ = rs[near[np.argmin(np.abs(rf[near] - b))]][1] if len(near) else "other"
            found.append(((b - a) / FPS, typ, cov >= 0.5))
            err_s.append(nearest(np.array([a]), ps)[0])
            err_e.append(nearest(np.array([b]), pe)[0])
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print("## DFL simulated, control reading (E7, cost chosen excluding the evaluated fold)\n")
    print(f"Frame level, D: precision {prec:.3f}, recall {rec:.3f}, F1 {2 * prec * rec / (prec + rec):.3f}\n")
    for name, e in (("stoppage start", err_s), ("stoppage end", err_e)):
        e = np.array(e)
        ok = np.isfinite(e)
        print(f"- {name}: n {len(e)}, unmatched within 3 s {1 - ok.mean():.2f}, mean error "
              f"{np.mean(e[ok]):+.2f} s, median {np.median(e[ok]):+.2f} s, median abs "
              f"{np.median(np.abs(e[ok])):.2f} s")
    found = np.array(found, dtype=object)
    print("\n| stoppage duration | n | found (predicted D covers >= 50 %) |\n|---|---|---|")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = np.array([lo <= d < hi for d in found[:, 0]])
        hi_s = f"{hi:g}" if np.isfinite(hi) else "inf"
        print(f"| {lo:g}-{hi_s} s | {sel.sum()} | {np.mean(found[sel, 2].astype(bool)):.2f} |")
    print("\n| restart type | n | found |\n|---|---|---|")
    for typ in sorted(set(found[:, 1])):
        sel = found[:, 1] == typ
        print(f"| {typ} | {sel.sum()} | {np.mean(found[sel, 2].astype(bool)):.2f} |")

    if len(sys.argv) < 2:
        return
    lab = pd.read_csv(sys.argv[1])["label"].to_numpy()
    graded = np.zeros(len(lab), bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    graded[PUBLIC[0]:PUBLIC[1] + 1] = False
    print("\n## BAR-ATH, submitted file, no labels (frames 11185-18684 excluded)\n")
    print(f"Share of graded frames predicted D: {(lab[graded] == 'D').mean():.3f}\n")
    tags = [t for t in load_smart_tagging.load(GAME) if t["code"] == "SET PIECES"
            and not (PUBLIC[0] - 5 * FPS <= t["start_frame"] <= PUBLIC[1])]
    print("| set piece type (Smart Tagging) | n | overlaps predicted D (interval, 5 s back) |\n|---|---|---|")
    types = {}
    for t in tags:
        seg = lab[max(t["start_frame"] - 5 * FPS, 0):t["end_frame"]]
        types.setdefault(t["tags"].get("Type", "?"), []).append(bool((seg == "D").any()))
    for typ, v in sorted(types.items()):
        print(f"| {typ} | {len(v)} | {np.mean(v):.2f} |")
    bar = barath.load()
    L, W = bar.pitch
    x, y = bar.ball[:, 0], bar.ball[:, 1]
    out = np.isfinite(x) & ((np.abs(x) > L / 2 + 0.5) | (np.abs(y) > W / 2 + 0.5)) & graded
    eps = [(a, b) for a, b in runs(out) if b - a >= FPS // 2]
    miss = [(a, b) for a, b in eps if not (lab[max(a - TOL, 0):b + TOL] == "D").any()]
    print(f"\nVisible ball outside the lines (> 0.5 m, >= 0.5 s): {len(eps)} episodes; without "
          f"predicted D within 3 s: {len(miss)}.")


if __name__ == "__main__":
    main()
