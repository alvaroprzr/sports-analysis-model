"""Additions to 1b / 1c (external review, points 4 and 8), from saved predictions only:

1. B1, E0, E7 per match and mean in four conditions: simulated / clean DFL x flag / control
   reading, decoder cost chosen excluding the evaluated fold (scripts/xfold.py).
2. Signed boundary error of E7 (control reading, simulated DFL): for each true boundary of a type
   (possession start, possession end, end of a stoppage), the nearest predicted boundary of the
   same type within +-3 s, matched one to one greedily by distance; error = predicted - true (s).
   Reported by the duration of the true possession (starts and ends) and overall for stoppage
   ends, with the fraction of true boundaries left without a match.
3. Would-be changes of the flag by duration and decision (rule of src/poss/control.py), and how
   many reverted runs end in a stoppage (a touch of the other team just before the ball goes out).
   The control labels keep the flag's 25 Hz boundaries: a run is kept or merged whole, so every
   boundary of the control reading is a boundary of the flag; checked below frame by frame.
4. Two more ceilings of E7 (control reading): boundaries (each predicted possession matched to a
   true one, IoU >= 0.3, repainted on the true possession's exact frames) and short possessions of
   1-4 s perfect.

    python scripts/control_extra.py > experiments/control_extra.md
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import control_1c  # noqa: E402
import xfold  # noqa: E402
from poss import control, dataset, decode, evaluate  # noqa: E402
from poss.evaluate import pm  # noqa: E402

MODELS = {"B1": "sim", "E0": "b2e0_5f", "E7": "ens_e7"}
TOL_S = 3.0
BINS = evaluate.DURATION_BINS_S


def load(tag):
    return pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))


def decoded(tag):
    """Native codes per held-out set at the cost chosen excluding its fold."""
    _, chosen = xfold.scores(tag)
    out = {}
    for k, r in load(tag).items():
        c = chosen[xfold.fold_of(k)]
        out[k] = decode.to_native(decode.decode(r["p_alive"], r["p_home"], r["period"],
                                                (c, xfold.C_STOP, xfold.C_STOP)), r["near"], r["n"])
    return out


def boundaries(codes):
    """(frame, type) with type 'start', 'end', 'dead_end'; and spell durations for start/end."""
    t = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    prev, cur = codes[t - 1], codes[t]
    live = lambda x: (x == dataset.H) | (x == dataset.A)
    return {"start": t[live(cur) & (prev != dataset.X)],
            "end": t[live(prev) & (cur != dataset.X)],
            "dead_end": t[(prev == dataset.D) & live(cur)]}


def spell_len(codes, frames, at_start):
    """Duration (s) of the possession that starts (or ends) at each frame."""
    out = []
    for f in frames:
        lab = codes[f] if at_start else codes[f - 1]
        a = b = f if at_start else f - 1
        while b + 1 < len(codes) and codes[b + 1] == lab:
            b += 1
        while a - 1 >= 0 and codes[a - 1] == lab:
            a -= 1
        out.append((b - a + 1) / pm.FPS)
    return np.array(out)


def match(gt_f, pr_f):
    tol = TOL_S * pm.FPS
    pairs = sorted((abs(int(p) - int(g)), i, j) for i, g in enumerate(gt_f)
                   for j in np.flatnonzero(np.abs(pr_f - g) <= tol) for p in [pr_f[j]])
    used_g, used_p, err = set(), set(), {}
    for _, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        err[i] = (int(pr_f[j]) - int(gt_f[i])) / pm.FPS
    return err


def boundary_ceiling(gt, pred):
    runs = pm.valid_runs(gt)
    g = pm.possession_spells(gt, runs)
    p = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    new = pred.copy()
    used_g, used_p = set(), set()
    for iou, i, j in pm.spell_overlaps(g, p):  # same team only, best IoU first
        if iou < 0.3:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        a, b, lab = g[i]
        new[a:b] = lab
    return new


def main():
    print("## 1. Four conditions (cost chosen excluding the evaluated fold)\n")
    res = {}
    for name, tag in MODELS.items():
        for reading, t in (("flag", tag), ("control", xfold.with_control(tag))):
            res[(name, reading)] = xfold.scores(t)[0]
    mids = b1_dev.SPLIT["dfl"]["dev"]
    for cond in ("sim", "clean"):
        print(f"### DFL {cond}\n")
        print("| match | " + " | ".join(f"{n} {r}" for n in MODELS for r in ("flag", "control")) + " |")
        print("|---|" + "---|" * (2 * len(MODELS)))
        for mid in mids + ["mean"]:
            cells = []
            for n in MODELS:
                for r in ("flag", "control"):
                    s = res[(n, r)]
                    v = (np.mean([s[f"dfl_{cond}_{m}"] for m in mids]) if mid == "mean"
                         else s[f"dfl_{cond}_{mid}"])
                    cells.append(f"{v:.3f}")
            print(f"| {mid} | " + " | ".join(cells) + " |")
        print()

    print("## 2. Signed boundary error of E7 (control reading, simulated DFL)\n")
    tag = xfold.with_control("ens_e7")
    dec = decoded(tag)
    pred = load(tag)
    acc = {"start": [], "end": [], "dead_end": []}
    for k, r in pred.items():
        if not k.startswith("dfl_sim"):
            continue
        gb, pb = boundaries(r["gt"]), boundaries(dec[k])
        for typ in acc:
            err = match(gb[typ], pb[typ])
            dur = (spell_len(r["gt"], gb[typ], typ != "end") if typ != "dead_end"
                   else np.zeros(len(gb[typ])))
            for i in range(len(gb[typ])):
                acc[typ].append((dur[i], err.get(i, np.nan)))
    print(f"Matched within +-{TOL_S:g} s; error = predicted - true; negative = early.\n")
    print("| boundary | true possession | n | unmatched | mean error s | median s | median abs s |")
    print("|---|---|---|---|---|---|---|")
    for typ, rows in acc.items():
        rows = np.array(rows)
        groups = ([("all", np.ones(len(rows), bool))] if typ == "dead_end" else
                  [(f"{BINS[i]:g}-{BINS[i + 1]:g} s", (rows[:, 0] >= BINS[i]) & (rows[:, 0] < BINS[i + 1]))
                   for i in range(len(BINS) - 1)])
        for gname, sel in groups:
            e = rows[sel, 1]
            ok = np.isfinite(e)
            if not len(e):
                continue
            print(f"| {typ} | {gname} | {len(e)} | {1 - ok.mean():.2f} | "
                  f"{np.mean(e[ok]):+.2f} | {np.median(e[ok]):+.2f} | {np.median(np.abs(e[ok])):.2f} |")

    print("\n## 3. Would-be changes of the flag, by duration and decision\n")
    log, exact = [], True
    for mid in mids:
        m = b1_dev.load("dfl", mid)
        ev, ko = control.load_events(mid)
        sh = control.fit_shift(m, ev, ko)
        team, _ = control.relabel(m, ev, ko, sh, log=log)
        gf = dataset.codes(m.alive, m.team, m.period)
        gc = dataset.codes(m.alive, team, m.period)
        bf = set(np.flatnonzero(gf[1:] != gf[:-1]))
        bc = set(np.flatnonzero(gc[1:] != gc[:-1]))
        exact &= bc <= bf
    log = np.array([(d, ["by_event", "by_duration", "by_passes", "reverted"].index(dec), st)
                    for d, dec, st in log])
    print("| run duration | n | kept by event | kept by >= 3 s | kept by passes | reverted | "
          "reverted and ending in a stoppage |\n|---|---|---|---|---|---|---|")
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, np.inf)):
        sel = (log[:, 0] >= lo) & (log[:, 0] < hi)
        n = int(sel.sum())
        c = [int((log[sel, 1] == j).sum()) for j in range(4)]
        st = int(((log[:, 1] == 3) & (log[:, 2] == 1) & sel).sum())
        print(f"| {lo:g}-{hi:g} s | {n} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {st} |")
    print(f"\nEvery boundary of the control reading is a boundary of the flag (25 Hz, exact "
          f"frame): {'yes' if exact else 'NO'}. Event times only decide whether a run is kept; "
          "the event-to-tracking shift per half is fitted automatically (control.fit_shift), "
          "with no manual adjustment.\n")

    print("## 4. More ceilings of E7 (control reading, cost chosen excluding the fold)\n")
    print("| ceiling | DFL sim | SC live | combined |\n|---|---|---|---|")
    base = {k: dec[k] for k in pred}
    for name in ("none", "boundaries", "short 1-4 s"):
        sc = {}
        for k, r in pred.items():
            if name == "none":
                c = base[k]
            elif name == "boundaries":
                c = boundary_ceiling(r["gt"], base[k])
            else:
                c = base[k]
            sc[k] = c
        if name == "short 1-4 s":
            # perfect probabilities on the frames of true possessions of 1 to 4 s, re-decoded
            orc = {}
            for k, r in pred.items():
                r = dict(r)
                g = r["gt"][r["near"]]
                fps = 10 if k.startswith("sc_") else pm.FPS  # SkillCorner tracking is 10 fps
                m4 = control_1c.short_mask(r["gt"], r["near"], fps, 4.0)
                m1 = control_1c.short_mask(r["gt"], r["near"], fps, 1.0)
                sm = m4 & ~m1 & (g < dataset.X)
                pa, ph = r["p_alive"].copy(), r["p_home"].copy()
                live = sm & (g < dataset.D)
                ph[live] = np.where(g[live] == dataset.H, 0.999, 0.001)
                if k.startswith("dfl"):
                    pa[sm] = np.where(g[sm] != dataset.D, 0.999, 0.001)
                r["p_alive"], r["p_home"] = pa, ph
                orc[k] = r
            pickle.dump(orc, open(os.path.join(b1_dev.CACHE, f"pred_{tag}_or_s14.pkl"), "wb"))
            sc = decoded(f"{tag}_or_s14")
        s = {k: (evaluate.full_scores(pred[k]["gt"], v)["spellF1"] if k.startswith("dfl")
                 else evaluate.live_scores(pred[k]["gt"], v)["spellF1_live"]) for k, v in sc.items()}
        d, scl, cmb = xfold.summary(s)
        print(f"| {name} | {d:.3f} | {scl:.3f} | {cmb:.3f} |", flush=True)


if __name__ == "__main__":
    main()
