"""B', last attempt: does arm g (scripts/alive_dev.py: arm c plus DFL training copies with the ball
gaps at positions independent of the game) replace E7's P(alive)? External data only. Rule fixed on
13 November, before arm g was trained; it replaces the rule of scripts/alive_select.py.

Why the rule changed. The old rule asked that the SkillCorner live score did not fall. That score
drops every frame labelled dead before the possession spells are built, so a predicted stoppage can
only cut spells there and never earns anything: every arm that found more stoppages lost on it, and
so does the frozen E7 itself with another decoder (experiments/alive_posthoc.md). The condition is
replaced by one on what it was meant to protect, that the predicted stoppages are real (2), and by
the regime in which arm c failed (3).

Single candidate: g. Arm d (recalibration of E7) does not enter: on the figures already measured
(experiments/alive_select.md) it gains a third of what c gains on SkillCorner and loses on DFL with
the control reading by 2 paired SE. Decoder of g: chosen as in scripts/alive_select.py (zone score
over DFL with the dead ball hidden, both readings, and SkillCorner metric 1, EXCLUDING the evaluated
fold); the stress copies take no part in the choice. Reference: the frozen E7, submitted decoder.

Adopt g only if
  1. SkillCorner full spellF1, metric 1 (frames with fewer than 14 detected players neutralised),
     is better than the frozen E7's by more than 2 paired SE (17 matches);
  2. at least 0.85 of its predicted stoppages on SkillCorner are real (definition of check T1 of
     scripts/alive_checks.py);
  3. on DFL with the dead ball hidden and the ball gaps at positions independent of the game (the
     copies of check T2, about 44 % of live play without the ball) it is not worse by more than 1
     paired SE, flag and control readings;
  4. on DFL with the dead ball hidden and the usual gaps it is not worse by more than 1 paired SE,
     flag and control readings.
The SkillCorner live score is reported and does not decide. Neither the
reserved test nor any frame of the provided match is read. If g does not pass, B' is closed.

    POSS_SIM=ext_dr python scripts/alive_select2.py > experiments/alive_select2.md
The last line is the decision as JSON.
"""

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import alive_checks as chk  # noqa: E402
import alive_dev  # noqa: E402
import alive_select as sel  # noqa: E402
import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
import xfold  # noqa: E402
from poss import dataset, evaluate  # noqa: E402

ARMS = ("frozen", "c", "g")
sel.NAMES["g"] = "g: c + gaps independent of the game"
D, X = dataset.D, dataset.X
FPS_SC = chk.FPS_SC


def load():
    sets, _ = sel.load_sets()
    p = pickle.load(open(alive_dev.pred_path("g"), "rb"))
    for k, s in sets.items():
        assert len(p[k]) == len(s["period"]), k
        s["pa"]["g"] = p[k]
    raw = pickle.load(open(alive_dev.STRESS, "rb"))
    stress = {}
    for mid, _ in b1_dev.folds():
        k = f"dfl_stress_{mid}"
        r = raw[k]
        assert len(p[k]) == len(r["period"]), k
        stress[k] = dict(period=r["period"], near=r["near"], n=r["n"], gt=r["gt"], ball=r["ball"],
                         gt_ctl=b2_dev.control_codes(mid)[0], ph=r["ph"], pa=dict(r["pa"], g=p[k]))
    return sets, stress


def stoppages(sets, arm, chosen):
    """Check T1: predicted stoppages on SkillCorner, the real ones, restart events, those found."""
    tot = np.zeros(4)
    for k, s in sets.items():
        if not k.startswith("sc_"):
            continue
        R = chk.restarts(k[3:])
        R = R[R < s["n"]]
        pred = sel.native(k, s, arm, *chosen[xfold.fold_of(k)], exact=True)
        for a, b in chk.runs((pred == D) & (s["gt"] < X)):
            if b - a >= 2 * FPS_SC and (~s["low"][a:b]).mean() > 0.5:
                real = (s["gt"][a:b] == D).any() or ((R >= a) & (R <= b + 5 * FPS_SC)).any()
                tot += [1, real, 0, 0]
        for r in R:
            tot += [0, 0, 1, (pred[max(r - 5 * FPS_SC, 0):r + 1] == D).any()]
    return tot


def stress_scores(stress, arm, chosen):
    out = {}
    for k, s in stress.items():
        pred = sel.native(k, s, arm, *chosen[xfold.fold_of(k)], exact=True)
        f, c = sel.main_scores(k, s, pred)
        out[k] = dict(flag=f, control=c, frames=sel.frame_counts(s, pred, s["gt"] < X))
    return out


def main():
    assert b1_dev.SIM == "ext_dr"
    sets, stress = load()
    dk = [k for k in sets if k.startswith("dfl")]
    sk = [k for k in sets if k.startswith("sc_")]
    tk = list(stress)
    res = {arm: sel.evaluate_arm(sets, arm) for arm in ARMS}
    st = {arm: stress_scores(stress, arm, res[arm][1]) for arm in ARMS}
    t1 = {arm: stoppages(sets, arm, res[arm][1]) for arm in ARMS}
    mean = lambda r, keys, f: np.mean([r[k][f] for k in keys])
    f = lambda d: f"{d[0]:+.4f} ({d[1]:.4f})"

    print("B', last attempt: arm g against the frozen E7 under the rule of scripts/alive_select2.py, fixed "
          "before g was trained. Five folds; held-out sets: 17 SkillCorner, 5 DFL with the dead ball hidden, "
          "and the same 5 DFL with the ball gaps at positions independent of the game (stress). Decoder "
          "chosen excluding the evaluated fold, without the stress copies. Arm c is shown for reference.\n")
    print("## Scores (spellF1)\n")
    print("| P(alive) | decoder per fold (c_team, c_stop) | SC full, metric 1 | DFL stress, flag | DFL stress, "
          "control | DFL hidden, flag | DFL hidden, control | SC live (does not decide) |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        r, chosen, _ = res[arm]
        dec = ", ".join(f"({c}, {cs:g})" for c, cs in chosen.values())
        print(f"| {sel.NAMES[arm]} | {dec} | {mean(r, sk, 'm1'):.4f} | {mean(st[arm], tk, 'flag'):.4f} | "
              f"{mean(st[arm], tk, 'control'):.4f} | {mean(r, dk, 'flag'):.4f} | {mean(r, dk, 'control'):.4f} "
              f"| {mean(r, sk, 'live'):.4f} |")

    print("\n## Predicted stoppages on SkillCorner (>= 2 s, >= 14 detected players on most frames)\n")
    print("| P(alive) | predicted stoppages per match | real (labelled dead or a restart event) "
          "| restart events with predicted D in the 5 s before |")
    print("|---|---|---|---|")
    for arm in ARMS:
        n, real, nr, found = t1[arm]
        print(f"| {sel.NAMES[arm]} | {n / len(sk):.1f} | {real / n:.3f} | {found / nr:.3f} |")

    print("\n## The rule: arm minus the frozen E7, paired by match (SE)\n")
    print("| P(alive) | 1. SC full, metric 1 (> 2 SE) | 2. real stoppages (>= 0.85) | 3. stress flag (>= -1 SE) "
          "| 3. stress control (>= -1 SE) | 4. hidden flag (>= -1 SE) | 4. hidden control (>= -1 SE) "
          "| SC live (does not decide) | passes |")
    print("|---|---|---|---|---|---|---|---|---|")
    ref, ref_st = res["frozen"][0], st["frozen"]
    verdict = {}
    for arm in ("c", "g"):
        r = res[arm][0]
        d1 = sel.paired(r, ref, sk, "m1")
        share = t1[arm][1] / t1[arm][0]
        d3 = [sel.paired(st[arm], ref_st, tk, x) for x in ("flag", "control")]
        d4 = [sel.paired(r, ref, dk, x) for x in ("flag", "control")]
        conds = [d1[0] > 2 * d1[1], share >= 0.85, all(d[0] >= -d[1] for d in d3),
                 all(d[0] >= -d[1] for d in d4)]
        verdict[arm] = conds
        print(f"| {sel.NAMES[arm]} | {f(d1)} | {share:.3f} | {f(d3[0])} | {f(d3[1])} | {f(d4[0])} | {f(d4[1])} | "
              f"{f(sel.paired(r, ref, sk, 'live'))} | {'YES' if all(conds) else 'no'} |")
    ok = all(verdict["g"])
    failed = [str(i + 1) for i, c in enumerate(verdict["g"]) if not c]
    print("\n**Decision by the rule: " + ("arm g ADOPTED**" if ok else
                                         f"arm g fails condition {', '.join(failed)}; the frozen E7 stays and "
                                         "B' is closed**"))
    print("\n(The row of c only applies the new conditions to the earlier arm for reference; c was not a "
          "candidate.)")

    print("\n## Stoppages, frame level, after decoding\n")
    print("| P(alive) | SC, >= 14 detected: D recall | D precision | predicted D among live | DFL stress: "
          "D recall | D precision | predicted D among live | DFL hidden: D recall | D precision "
          "| predicted D among live |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        cells = []
        for fr in (sum(res[arm][0][k]["frames"] for k in sk), sum(st[arm][k]["frames"] for k in tk),
                   sum(res[arm][0][k]["frames"] for k in dk)):
            cells += [f"{fr[0] / fr[1]:.3f}", f"{fr[0] / max(fr[2], 1):.3f}", f"{fr[3] / fr[4]:.3f}"]
        print(f"| {sel.NAMES[arm]} | " + " | ".join(cells) + " |")

    out = {"adopted": "g" if ok else None}
    c, cs = res["g"][2]
    print(f"\nDecoder of g on all folds: ({c}, {cs:g})")
    if ok:
        out.update(costs=[c, cs, cs])
    print("\n" + json.dumps(out))


if __name__ == "__main__":
    main()
