"""Two more checks of arm c of B' (scripts/alive_dev.py), external data only. The rule of
scripts/alive_select.py rejected every arm: c improves the full scores of both sources and loses on
SkillCorner live by more than 1 paired SE. Before measuring anything else, on 13 November, it was
fixed that c would be adopted as a documented amendment only if both checks pass:

T1  Spurious stoppages on real tracking (17 SkillCorner development matches, out of fold). A
    predicted stoppage is a run of predicted D of at least 2 s with at least 14 detected players on
    most of its frames. It is real if it overlaps a frame labelled dead or a restart event of
    SkillCorner's dynamic events (game_interruption_before, events within 5 s merged) falls in
    [start, end + 5 s]. Pass: at least 0.85 of the predicted stoppages are real.
T2  Many ball gaps in live play (5 DFL development matches, held-out fold). The ball gaps of the
    held-out degradation at positions independent of the game (about 44 % of live play without the
    ball) and the dead ball hidden. Pass: the spellF1 of c is not worse than the frozen E7's by
    more than 1 paired SE, with the flag and with the control reading.

Known when the checks were fixed: SkillCorner's labels and restart events agree (95 % of the
restarts have a labelled-dead frame in the 5 s before), and at frame level c has D precision 0.908
and predicts D on 2.3 % of SkillCorner's live frames with at least 14 detected players.

Decoders: c at the costs chosen without each fold in experiments/alive_select.md; frozen E7 (3, 8).

    POSS_SIM=ext_dr python scripts/alive_checks.py > experiments/alive_checks.md
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import alive_dev  # noqa: E402
import alive_select as sel  # noqa: E402
import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
import xfold  # noqa: E402
from poss import dataset, evaluate  # noqa: E402

RAW = os.path.join(ROOT, "data", "external", "skillcorner_raw", "repo", "data", "matches")
DEC_C = [(3, 4.0), (3, 4.0), (3, 6.0), (2, 6.0), (2, 6.0)]  # experiments/alive_select.md, per fold
DEC_F = (3, 8.0)
FPS_SC = 10
D, X = dataset.D, dataset.X


def restarts(mid):
    df = pd.read_csv(os.path.join(RAW, mid, f"{mid}_dynamic_events.csv"), low_memory=False,
                     usecols=["frame_start", "game_interruption_before"])
    out = []
    for f in sorted(df.loc[df["game_interruption_before"].notna(), "frame_start"].astype(int)):
        if not out or f - out[-1] > 5 * FPS_SC:
            out.append(f)
    return np.array(out)


def runs(mask):
    t = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(t[::2], t[1::2]))


def t1(sets):
    tot = {a: np.zeros(4) for a in ("frozen", "c")}  # predicted stoppages, real, restarts, restarts found
    for k, s in sets.items():
        if not k.startswith("sc_"):
            continue
        R = restarts(k[3:])
        R = R[R < s["n"]]
        ok = s["gt"] < X
        for arm, dec in (("frozen", DEC_F), ("c", DEC_C[xfold.fold_of(k)])):
            pred = sel.native(k, s, arm, *dec, exact=True)
            for a, b in runs((pred == D) & ok):
                if b - a >= 2 * FPS_SC and (~s["low"][a:b]).mean() > 0.5:
                    real = (s["gt"][a:b] == D).any() or ((R >= a) & (R <= b + 5 * FPS_SC)).any()
                    tot[arm] += [1, real, 0, 0]
            for r in R:
                tot[arm] += [0, 0, 1, (pred[max(r - 5 * FPS_SC, 0):r + 1] == D).any()]
    print("## T1. Predicted stoppages on SkillCorner (>= 2 s, >= 14 detected players on most frames)\n")
    print("| P(alive) | predicted stoppages per match | real (labelled dead or a restart event) "
          "| restart events with predicted D in the 5 s before |")
    print("|---|---|---|---|")
    for arm, (n, real, nr, found) in tot.items():
        print(f"| {sel.NAMES[arm]} | {n / 17:.1f} | {real / n:.3f} | {found / nr:.3f} ({nr / 17:.1f} per match) |")
    share = tot["c"][1] / tot["c"][0]
    ok = share >= 0.85
    print(f"\nT1: {share:.3f} of the predicted stoppages of c are real -> " + ("PASS (>= 0.85)" if ok else "FAIL (< 0.85)"))
    return ok


def t2():
    pred = pickle.load(open(alive_dev.STRESS, "rb"))
    rows = {a: {} for a in ("frozen", "c")}
    frames = {a: np.zeros(5) for a in rows}
    vis = []
    for i, (mid, _) in enumerate(b1_dev.folds()):
        k = f"dfl_stress_{mid}"
        s = pred[k]
        ctl = b2_dev.control_codes(mid)[0]
        ok = s["gt"] < X
        y = s["gt"][s["near"]]
        live = (y < D) & (s["period"] > 0)
        vis.append((s["ball"][live, 4] < 0.5).mean())
        for arm, dec in (("frozen", DEC_F), ("c", DEC_C[i])):
            p = sel.native(k, s, arm, *dec, exact=True)
            rows[arm][k] = dict(flag=evaluate.full_scores(s["gt"], p)["spellF1"],
                                control=evaluate.full_scores(ctl, p)["spellF1"])
            dead, pd_ = ok & (s["gt"] == D), ok & (p == D)
            frames[arm] += [(dead & pd_).sum(), dead.sum(), pd_.sum(), (pd_ & ~dead).sum(), (ok & ~dead).sum()]
    keys = list(rows["c"])
    print("\n## T2. DFL with ball gaps independent of the game and the dead ball hidden\n")
    print(f"Live rows without the ball: {np.mean(vis):.3f} (mean of 5 matches).\n")
    print("| P(alive) | spellF1 flag | spellF1 control | D recall | D precision | predicted D among live |")
    print("|---|---|---|---|---|---|")
    for arm, r in rows.items():
        f = frames[arm]
        print(f"| {sel.NAMES[arm]} | {np.mean([r[k]['flag'] for k in keys]):.4f} | "
              f"{np.mean([r[k]['control'] for k in keys]):.4f} | {f[0] / f[1]:.3f} | {f[0] / f[2]:.3f} | {f[3] / f[4]:.3f} |")
    df, dc = sel.paired(rows["c"], rows["frozen"], keys, "flag"), sel.paired(rows["c"], rows["frozen"], keys, "control")
    ok = df[0] >= -df[1] and dc[0] >= -dc[1]
    print(f"\nT2: c minus frozen, flag {df[0]:+.4f} (SE {df[1]:.4f}), control {dc[0]:+.4f} (SE {dc[1]:.4f}) -> "
          + ("PASS (neither worse by more than 1 SE)" if ok else "FAIL"))
    return ok


def main():
    assert b1_dev.SIM == "ext_dr"
    print("Checks T1 and T2 of arm c of B', fixed on 13 November before measuring (scripts/alive_checks.py).\n")
    sets, _ = sel.load_sets()
    a = t1(sets)
    b = t2()
    print("\n**" + ("Both pass: arm c is adopted as a documented amendment" if a and b
                    else "Not both pass: arm c is not adopted; the frozen E7 stays") + "**")


if __name__ == "__main__":
    main()
