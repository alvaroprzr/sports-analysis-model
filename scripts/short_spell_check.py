"""Short possessions of the submission on the held-out development sets (external data only, no
training, nothing chosen). Definitions fixed on 14 November before any result:
  short possession   a maximal run of H or of A in the labels shorter than 3 s;
  evidence           P(home | alive) is on the side of its team on at least half of the model rows
                     inside it;
  disappears         after the decoder its team has fewer than half of its frames.
Criterion for trying a team-switch cost that adapts to E5's change head: at least 25 % of the short
possessions with evidence disappear in at least one provider (DFL with the flag).

    POSS_SIM=ext_dr python scripts/short_spell_check.py > experiments/short_spell_check.md
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import alive_select as sel  # noqa: E402
import alive_select2 as sel2  # noqa: E402
import b1_dev  # noqa: E402
import xfold  # noqa: E402
from poss import dataset  # noqa: E402

H, A = dataset.H, dataset.A
SHORT_S = 3.0


def spells(gt):
    cut = np.flatnonzero(np.diff(gt)) + 1
    a = np.r_[0, cut]
    b = np.r_[cut, len(gt)]
    return [(i, j, gt[i]) for i, j in zip(a, b) if gt[i] in (H, A)]


def main():
    assert b1_dev.SIM == "ext_dr"
    sets, _ = sel2.load()
    chosen = sel.evaluate_arm(sets, "g")[1]
    tot = {}
    for k, s in sets.items():
        pred = sel.native(k, s, "g", *chosen[xfold.fold_of(k)], exact=True)
        fps = 25 if k.startswith("dfl") else 10
        for name, gt in ([("DFL, flag", s["gt"]), ("DFL, second reading", s["gt_ctl"])] if k.startswith("dfl")
                         else [("SkillCorner", s["gt"])]):
            t = tot.setdefault(name, np.zeros(6))
            for i, j, team in spells(gt):
                long_ = (j - i) / fps >= SHORT_S
                kept = (pred[i:j] == team).mean() >= 0.5
                if long_:
                    t += [0, 0, 0, 0, 1, kept]
                    continue
                r = np.flatnonzero((s["near"] >= i) & (s["near"] < j))
                ev = len(r) > 0 and ((s["ph"][r] > 0.5) == (team == H)).mean() >= 0.5
                t += [1, kept, ev, ev and not kept, 0, 0]
    print("Short possessions (< 3 s) of the submission on the held-out development sets "
          "(scripts/short_spell_check.py; definitions and criterion fixed before any result).\n")
    print("| labels | short possessions | kept by the decoder | with evidence in the probabilities "
          "| with evidence, lost by the decoder | share of those with evidence | possessions >= 3 s kept |")
    print("|---|---|---|---|---|---|---|")
    shares = {}
    for name, (n, kept, ev, lost, nl, kl) in tot.items():
        shares[name] = lost / ev
        print(f"| {name} | {int(n)} | {kept / n:.3f} | {ev / n:.3f} | {int(lost)} | {lost / ev:.3f} | {kl / nl:.3f} |")
    ok = max(shares["DFL, flag"], shares["SkillCorner"]) >= 0.25
    print("\nCriterion (at least 0.25 of the short possessions with evidence lost, in at least one provider): "
          + ("met" if ok else "not met; the adaptive switch cost is not tried"))


if __name__ == "__main__":
    main()
