"""Choice of the simulator arm (rule fixed on 06/11 before the results):

Each arm is a B1 trained on 5 folds (DFL degraded by that arm, SkillCorner team labels) and scored
on the 14 SkillCorner development matches not used to calibrate the external simulator, at three
fixed levels: as they are, and with extra ball gaps down to 70 % and 40 % of visible ball
(b1_dev.SC_LEVELS). Per arm the team-switch cost is chosen by the zone over (SkillCorner as is,
clean DFL). The arm with the best mean of the three levels wins, provided its SkillCorner score as
is is not more than 1 paired SE below the best arm's.

    python scripts/sim_ext_select2.py TAG [TAG ...] > experiments/sim_ext_select2.md
"""

import sys

import numpy as np

import b1_compare
import b1_dev
import sim_ext_calibrate

CAL_SC = set(b1_dev.SPLIT["skillcorner"]["dev"][:sim_ext_calibrate.TARGET_SC])
DEV_SC = [s for s in b1_dev.SPLIT["skillcorner"]["dev"] if s not in CAL_SC]
LEVELS = ["sc", "scv70", "scv40"]


def main():
    res = {}
    for tag in sys.argv[1:]:
        g = {c: b1_compare.per_match(tag, (c, b1_compare.ZONE_C_STOP, b1_compare.ZONE_C_STOP))
             for c in b1_compare.ZONE_C_TEAM}
        k0 = g[b1_compare.ZONE_C_TEAM[0]]
        for lv in LEVELS:
            used = sorted(k.split("_", 1)[1] for k in k0 if k.split("_", 1)[0] == lv
                          and k.split("_", 1)[1] not in CAL_SC)
            assert used == sorted(DEV_SC), (tag, lv, used)  # exactly the 14, none of the 3
        sc = lambda c, lv: {s: g[c][f"{lv}_{s}"] for s in DEV_SC}
        cl = lambda c: np.mean([v for k, v in g[c].items() if k.startswith("dfl_clean")])
        rows = [(c, np.mean(list(sc(c, "sc").values())), cl(c)) for c in g]
        bs, bc = max(r[1] for r in rows), max(r[2] for r in rows)
        c = rows[int(np.argmin([max(bs - r[1], bc - r[2]) for r in rows]))][0]
        res[tag] = (c, {lv: sc(c, lv) for lv in LEVELS}, cl(c))
    print("| arm | c_team | SC as is | SC 70 % ball | SC 40 % ball | mean of 3 | clean DFL (control) |")
    print("|---|---|---|---|---|---|---|")
    for tag, (c, s, cl) in res.items():
        m = [np.mean(list(s[lv].values())) for lv in LEVELS]
        print(f"| {tag} | {c} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {np.mean(m):.3f} | {cl:.3f} |")
    best_as_is = max(res, key=lambda t: np.mean(list(res[t][1]["sc"].values())))
    ref = res[best_as_is][1]["sc"]
    print(f"\nBest arm on SkillCorner as is: {best_as_is}. Paired difference to it (14 matches):\n")
    eligible = []
    for tag, (c, s, _) in res.items():
        d = np.array([s["sc"][k] - ref[k] for k in DEV_SC])
        se = d.std(ddof=1) / np.sqrt(len(d)) if tag != best_as_is else 0.0
        ok = tag == best_as_is or d.mean() >= -se
        eligible.append(tag) if ok else None
        print(f"- {tag}: {d.mean():+.4f} (SE {se:.4f}) -> {'eligible' if ok else 'excluded'}")
    win = max(eligible, key=lambda t: np.mean([np.mean(list(res[t][1][lv].values())) for lv in LEVELS]))
    print(f"\nChosen: {win}")


if __name__ == "__main__":
    main()
