"""Choose the intensity of the external simulator by B1's downstream score on SkillCorner (real
tracking defects), leaving out the SkillCorner matches the simulator was calibrated on. The DFL
simulated score is not used to choose (it is measured with each simulator itself); clean DFL is
reported as a control. Rule fixed before the results: per variant, the team-switch cost is chosen
by the zone over (SkillCorner live without the calibration matches, clean DFL); the variant with the
best SkillCorner live score there wins.

    python scripts/sim_ext_select.py TAG [TAG ...] > experiments/sim_ext_select.md
"""

import sys

import numpy as np

import b1_compare
import b1_dev
import sim_ext_calibrate

CAL_SC = set(b1_dev.SPLIT["skillcorner"]["dev"][:sim_ext_calibrate.TARGET_SC])


def main():
    print(f"SkillCorner matches left out (simulator calibration): {', '.join(sorted(CAL_SC))}\n")
    print("| arm | c_team | SC live (dev, without the calibration matches) | n SC | DFL clean | "
          "DFL simulated (own simulator) |")
    print("|---|---|---|---|---|---|")
    best = None
    dev_sc = set(b1_dev.SPLIT["skillcorner"]["dev"])
    for tag in sys.argv[1:]:
        keys = list(b1_compare.per_match(tag, (3, 6, 6)))
        used = [k[3:] for k in keys if k.startswith("sc_") and k[3:] not in CAL_SC]
        folds = sorted({i for i, (d, _) in enumerate(b1_dev.folds()) if f"dfl_sim_{d}" in keys})
        expected = [s for i in folds for s in b1_dev.folds()[i][1] if s not in CAL_SC]
        # guard: a different key format must not silently disable the exclusion
        assert sorted(used) == sorted(expected) and set(used) <= dev_sc and not set(used) & CAL_SC, \
            (tag, used, expected)
        rows = []
        for c in b1_compare.ZONE_C_TEAM:
            s = b1_compare.per_match(tag, (c, b1_compare.ZONE_C_STOP, b1_compare.ZONE_C_STOP))
            sc = np.mean([v for k, v in s.items() if k.startswith("sc_") and k[3:] not in CAL_SC])
            cl = np.mean([v for k, v in s.items() if k.startswith("dfl_clean")])
            sim = np.mean([v for k, v in s.items() if k.startswith("dfl_sim")])
            rows.append((c, sc, cl, sim))
        bs, bc = max(r[1] for r in rows), max(r[2] for r in rows)
        c, sc, cl, sim = rows[int(np.argmin([max(bs - r[1], bc - r[2]) for r in rows]))]
        print(f"| {tag} | {c} | {sc:.3f} | {len(used)} | {cl:.3f} | {sim:.3f} |")
        if best is None or sc > best[1]:
            best = (tag, sc)
    print(f"\nChosen: {best[0]}")


if __name__ == "__main__":
    main()
