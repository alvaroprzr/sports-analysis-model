"""Decoder of the submission, chosen again on the out-of-fold development predictions of the models
of the chosen simulator (nothing carried over from earlier models). Policy fixed before any of
these predictions was scored (12/11):

1. Decoder: team-switch cost c_team, stoppage cost c_stop (entering and leaving D) and minimum
   possession duration k (only if the minimum duration was adopted, experiments/f4d_rule.md).
2. F5, ball out of the pitch (law of the game: a ball wholly over a line is out of play): applied
   with (c_team, min_out_m, min_s) chosen on development, unless some source (DFL sim, SkillCorner
   live) gets worse by more than 2 paired SE in either reading of the DFL labels (a guard against
   implementation errors, not a proof of improvement).
   F5 is a law of the game with an imperfect computational detection, not a statistically
   demonstrated improvement; the detection is also checked for systematic errors in each source
   (share of the rows it marks that are truly stopped, by line, after a gap of the ball, per
   match; WARNING if a cell with >= 50 rows is below 0.5).
3. Boundary shifts (src/poss/shift.py, the same shift for every boundary of a type: team change,
   start of a stoppage, end of a stoppage; each in SHIFT_GRID seconds): a heuristic, not a law of
   the game, so the strict rule: the combined score averaged over the two readings better by more
   than 2 paired SE and no source worse by more than 1 paired SE in either reading. If not, left
   out.

Every step is evaluated with its parameters chosen by the zone score over three sources (DFL sim
with the flag, with the control reading, SkillCorner live) EXCLUDING the evaluated fold; the
reference of each step is the decoder as adopted so far, with its team-switch cost chosen the same
way. The parameters of an adopted step for the submission are chosen on all folds; later steps
keep the earlier steps' parameters at those values and choose c_team again. Ties go to the first
value of each grid (c_stop 6, the earlier value; shift 0).

Also: the mean signed offset of the boundaries per type and source (predicted minus true, matched
within 2 s), before step 3.

    POSS_SIM=ext_dr python scripts/decoder_select.py TAG MINDUR(yes|no) > experiments/decoder_select.md
The last line is the decoder as JSON (for configs/b2_ens_dr.json).
"""

import itertools
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import f4_chain  # noqa: E402
import xfold  # noqa: E402
from f5_ballout import M_GRID, S_GRID, se  # noqa: E402
import b2_dev  # noqa: E402
from poss import ballout, dataset, decode, decode2, evaluate, shift, tokens  # noqa: E402

W = evaluate.SC_WEIGHT
C_GRID = xfold.C_GRID
CS_GRID = (6.0, 2.0, 4.0, 8.0, 12.0)
SHIFT_GRID = (0.0, -0.2, 0.2, -0.4, 0.4)
SRC = (("dfl_sim", "DFL sim"), ("sc_", "SC live"))


def native_score(k, r, codes10):
    c = decode.to_native(codes10, r["near"], r["n"])
    return (evaluate.full_scores(r["gt"], c)["spellF1"] if k.startswith("dfl")
            else evaluate.live_scores(r["gt"], c)["spellF1_live"])


class Sets:
    def __init__(self, tag):
        self.pf = f4_chain.load(tag)
        self.pc = f4_chain.load(xfold.with_control(tag))
        self.keys = [k for k in self.pf if k.startswith(("dfl_sim", "sc_"))]
        self.ball, self.pitch = {}, {}
        for k in self.keys:  # the same copies the models predicted (scripts/f4_infer.py)
            mid = k.split("_")[-1]
            if k.startswith("dfl_sim"):
                c, m = b2_dev.dfl_copy(mid, b1_dev.TEST_SEED, False), b1_dev.load("dfl", mid)
            else:
                c, m = b2_dev.sc_copy(mid, False), b1_dev.load("skillcorner", mid)
            self.ball[k], self.pitch[k] = c["ball"], (float(m.pitch[0]), float(m.pitch[1]))

    def p_alive(self, k, f5):
        pa = self.pf[k]["p_alive"]
        return pa if f5 is None else ballout.apply(pa, self.ball[k], *f5)

    def codes(self, k, d):
        r = self.pf[k]
        pa = self.p_alive(k, d["f5"])
        cs = d["c_stop"]
        c10 = decode2.decode(pa, r["p_home"], r["period"], (d["c_team"], cs, cs), k=d["k"])
        return shift.apply(c10, r["period"], d["shift"])

    def score(self, k, c10):
        """(flag, control) spellF1 of one set."""
        f = native_score(k, self.pf[k], c10)
        return f, (native_score(k, self.pc[k], c10) if k.startswith("dfl") else f)

    def grid(self, decs, reuse_decode=None):
        """{name: (flag scores, control scores)} for a list of (name, decoder dict)."""
        g = {}
        cache = {}
        for name, d in decs:
            sf, sc = {}, {}
            for k in self.keys:
                if reuse_decode:
                    base = reuse_decode(d)
                    key = (k, base)
                    if key not in cache:
                        cache[key] = self.codes(k, dict(d, shift={}))
                    r = self.pf[k]
                    c10 = shift.apply(cache[key], r["period"], d["shift"])
                else:
                    c10 = self.codes(k, d)
                sf[k], sc[k] = self.score(k, c10)
            g[name] = (sf, sc)
        return g


def zone3(g, keys):
    rows = []
    for name, (sf, sc) in g.items():
        rows.append((name, np.mean([sf[k] for k in keys if k.startswith("dfl_sim")]),
                     np.mean([sc[k] for k in keys if k.startswith("dfl_sim")]),
                     np.mean([sf[k] for k in keys if k.startswith("sc_")])))
    best = [max(r[i] for r in rows) for i in (1, 2, 3)]
    gaps = [max(best[0] - a, best[1] - b, best[2] - c) for _, a, b, c in rows]
    return rows[int(np.argmin(gaps))][0]


def xsel(g, keys):
    out = {"flag": {}, "control": {}}
    chosen = {}
    for i in range(b1_dev.N_FOLDS):
        own = [k for k in keys if xfold.fold_of(k) == i]
        name = zone3(g, [k for k in keys if xfold.fold_of(k) != i])
        chosen[i] = name
        for k in own:
            out["flag"][k], out["control"][k] = g[name][0][k], g[name][1][k]
    return out, chosen


def compare(new, ref, keys, strict):
    """Prints the comparison; returns True if adopted under the rule."""
    for name, s in (("reference", ref), ("candidate", new)):
        for reading in ("flag", "control"):
            d, sc, cmb = xfold.summary(s[reading])
            print(f"| {name} | {reading} | {d:.4f} | {sc:.4f} | {cmb:.4f} |")
    diff = {r: {k: new[r][k] - ref[r][k] for k in keys} for r in ("flag", "control")}
    dd = np.array([(diff["flag"][k] + diff["control"][k]) / 2 for k in keys if k.startswith("dfl_sim")])
    ds = np.array([(diff["flag"][k] + diff["control"][k]) / 2 for k in keys if k.startswith("sc_")])
    cmb = (dd.mean() + W * ds.mean()) / (1 + W)
    s_c = np.sqrt(se(dd) ** 2 + (W * se(ds)) ** 2) / (1 + W)
    print(f"\nCombined, mean of the two readings: {cmb:+.4f} (paired SE {s_c:.4f})")
    ok = cmb > 2 * s_c if strict else True
    for reading in ("flag", "control"):
        for prefix, label in SRC:
            x = np.array([v for k, v in diff[reading].items() if k.startswith(prefix)])
            ok &= x.mean() >= (-se(x) if strict else -2 * se(x))
            print(f"- {label}, {reading}: {x.mean():+.4f} (SE {se(x):.4f})")
    rule = ("strict: combined > 2 SE, no source worse than 1 SE" if strict
            else "law of the game: applied unless a source is worse than 2 SE")
    print(f"\nRule ({rule}): {'ADOPTED' if ok else 'not adopted'}\n")
    return bool(ok)


def f5_detection(S, q):
    """Systematic errors of the ball-out detection at (min_out_m, min_s) = q: of the 10 Hz rows it
    marks, the share truly stopped (D in the labels; the DFL control reading has the same D), per
    source, by line (goal line or touchline, from the ball position), by whether the run starts
    within 1 s after the ball was not visible, and per match."""
    print(f"Detection check at min_out_m {q[0]:g}, min_s {q[1]:g}: share of the marked rows that are "
          f"truly stopped (rows, share)\n")
    print("| source | all | goal line | touchline | after a gap (<= 1 s) | no gap | worst match |")
    print("|---|---|---|---|---|---|---|")
    warn = []
    for prefix, label in SRC:
        cells = {c: [0, 0] for c in ("all", "goal", "touch", "gap", "nogap")}
        per_match = []
        for k in S.keys:
            if not k.startswith(prefix):
                continue
            b = S.ball[k].astype(np.float64)
            out = ballout.out_rows(S.ball[k], *q)
            g10 = S.pf[k]["gt"][S.pf[k]["near"]]
            ok = out & (g10 < dataset.X)
            dead = g10 == dataset.D
            x = np.abs(b[:, 0]) * tokens.SCALE_POS[0]
            goal = x > S.pitch[k][0] / 2
            vis = b[:, ballout.VIS] > 0.5
            gap = np.zeros(len(out), bool)
            t = np.flatnonzero(np.diff(np.r_[0, out.astype(np.int8), 0]))
            for a, e in zip(t[::2], t[1::2]):
                gap[a:e] = (~vis[max(0, a - ballout.RATE):a]).any()
            for c, m in (("all", ok), ("goal", ok & goal), ("touch", ok & ~goal), ("gap", ok & gap),
                         ("nogap", ok & ~gap)):
                cells[c][0] += int(m.sum())
                cells[c][1] += int((m & dead).sum())
            if ok.sum():
                per_match.append(((ok & dead).sum() / ok.sum(), int(ok.sum()), k))
        fmt = lambda n, d: f"{n}, {d / n:.2f}" if n else "0"
        worst = min(per_match) if per_match else None
        print(f"| {label} | " + " | ".join(fmt(*cells[c]) for c in cells) + " | "
              + (f"{worst[0]:.2f} ({worst[1]} rows)" if worst else "-") + " |")
        for c, (n, d) in cells.items():
            if n >= 50 and d / n < 0.5:
                warn.append(f"{label} {c}")
        if worst and worst[1] >= 50 and worst[0] < 0.5:
            warn.append(f"{label} one match")
    print("\n" + (f"WARNING, possible systematic error: {', '.join(warn)}" if warn
                  else "No systematic error found (every cell with >= 50 rows at 0.5 or more).") + "\n")


def header():
    print("| decoder | reading | DFL sim | SC live | combined |\n|---|---|---|---|---|")


def with_c(base, c, **kw):
    return dict(base, c_team=c, **kw)


def main():
    tag, mindur = sys.argv[1], sys.argv[2] == "yes"
    S = Sets(tag)
    keys = S.keys
    k_grid = f4_chain.K_GRID if mindur else (1,)
    print(f"Predictions: {tag} (out of fold, {len(keys)} held-out sets: "
          f"{sum(k.startswith('dfl_sim') for k in keys)} DFL sim, {sum(k.startswith('sc_') for k in keys)} "
          f"SkillCorner). Minimum duration {'adopted' if mindur else 'not adopted'}.\n")
    empty = dict(f5=None, shift={})

    # 1. decoder
    print("## 1. Decoder (c_team, c_stop, minimum duration k)\n")
    decs = [((c, cs, k), dict(empty, c_team=c, c_stop=cs, k=k))
            for cs in CS_GRID for k in k_grid for c in C_GRID]
    g1 = S.grid(decs)
    est, ch = xsel(g1, keys)
    print(f"Excluding the evaluated fold, (c_team, c_stop, k) per fold: {list(ch.values())}\n")
    header()
    for reading in ("flag", "control"):
        d, sc, cmb = xfold.summary(est[reading])
        print(f"| chosen excluding the fold | {reading} | {d:.4f} | {sc:.4f} | {cmb:.4f} |")
    c, cs, k = zone3(g1, keys)
    print(f"\nFor the submission (all folds): c_team {c}, c_stop {cs:g}, k {k} rows.\n")
    dec = dict(empty, c_team=c, c_stop=cs, k=k)

    def ref_of(d):
        g = S.grid([(cc, with_c(d, cc)) for cc in C_GRID])
        return xsel(g, keys)

    # 2. F5
    print("## 2. Ball out of the pitch (F5, law of the game)\n")
    ref, ch_r = ref_of(dec)
    params = [(m, s) for m in M_GRID for s in S_GRID]
    g2 = S.grid([((cc, q), with_c(dec, cc, f5=q)) for q in params for cc in C_GRID])
    new, ch_n = xsel(g2, keys)
    print(f"Reference c_team per fold {list(ch_r.values())}; candidate (c_team, (min_out_m, min_s)) "
          f"per fold {list(ch_n.values())}\n")
    cc, q = zone3(g2, keys)
    f5_detection(S, q)
    header()
    if compare(new, ref, keys, strict=False):
        dec = with_c(dec, cc, f5=q)
        print(f"For the submission: c_team {cc}, min_out_m {q[0]:g}, min_s {q[1]:g}.\n")

    # 3. boundary shifts
    print("## 3. Boundary shifts (heuristic)\n")
    print("Mean signed offset of the boundaries, predicted minus true (s; matched within 2 s), "
          "decoder so far at its submission parameters:\n")
    print("| source | reading | " + " | ".join(f"{t} (n, mean, median)" for t in shift.TYPES) + " |")
    print("|---|---|" + "---|" * len(shift.TYPES))
    for prefix, label in SRC:
        for reading, P in (("flag", S.pf), ("control", S.pc)):
            if prefix == "sc_" and reading == "control":
                continue
            off = {t: [] for t in shift.TYPES}
            for k in keys:
                if not k.startswith(prefix):
                    continue
                r = P[k]
                nat = decode.to_native(S.codes(k, dec), r["near"], r["n"])
                fps = 25 if k.startswith("dfl") else 10
                for t, v in shift.signed_offsets(r["gt"], nat, fps).items():
                    off[t] += v
            cells = [f"{len(v)}, {np.mean(v):+.2f}, {np.median(v):+.2f}" if v else "0" for v in off.values()]
            print(f"| {label} | {reading} | " + " | ".join(cells) + " |")
    print("\n(SkillCorner labels have no stoppages of their own beyond out of play; its live score "
          "uses only the team boundaries.)\n")
    ref, ch_r = ref_of(dec)
    combos = list(itertools.product(SHIFT_GRID, repeat=len(shift.TYPES)))
    decs = [((cc, sh), with_c(dec, cc, shift=dict(zip(shift.TYPES, sh)))) for sh in combos for cc in C_GRID]
    g4 = S.grid(decs, reuse_decode=lambda d: d["c_team"])
    new, ch_n = xsel(g4, keys)
    print(f"Candidate (c_team, shifts {shift.TYPES} in s) per fold: {list(ch_n.values())}\n")
    header()
    if compare(new, ref, keys, strict=True):
        cc, sh = zone3(g4, keys)
        dec = with_c(dec, cc, shift=dict(zip(shift.TYPES, sh)))
        print(f"For the submission: c_team {cc}, shifts {dict(zip(shift.TYPES, sh))}.\n")

    out = {"costs": [dec["c_team"], dec["c_stop"], dec["c_stop"]], "min_possession_rows": dec["k"],
           "ball_out": ({"min_out_m": dec["f5"][0], "min_s": dec["f5"][1]} if dec["f5"] else None),
           "boundary_shift_s": dec["shift"] or None}
    print("## Decoder of the submission\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
