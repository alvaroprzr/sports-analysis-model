"""How SkillCorner enters B1's training: no SkillCorner (sim_nosc), SkillCorner for the team in
live play only (sim, the reserve submission), SkillCorner everywhere (sim_scfull), and the average
of the probabilities of sim and sim_nosc (no training). Each is scored with
b1_compare.zone_score and by possession duration at its own selected team-switch cost.

Caveat: an arm trained on SkillCorner is no longer judged by an independent SkillCorner, so the
zone score favours it; small differences are ties.

    python scripts/b1_sc_arms.py > experiments/b1_sc_arms.md
"""

import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_compare  # noqa: E402
import b1_dev  # noqa: E402


def average(a, b, tag):
    pa = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{a}.pkl"), "rb"))
    pb = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{b}.pkl"), "rb"))
    out = {}
    for k in pa:
        r = dict(pa[k])
        r["p_alive"] = (pa[k]["p_alive"] + pb[k]["p_alive"]) / 2
        r["p_home"] = (pa[k]["p_home"] + pb[k]["p_home"]) / 2
        out[k] = r
    pickle.dump(out, open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "wb"))


def main():
    average("sim", "sim_nosc", "sim_avg_nosc")
    arms = [("sim_nosc", "no SkillCorner"), ("sim", "SkillCorner team in live play (reserve)"),
            ("sim_scfull", "SkillCorner everywhere"),
            ("sim_avg_nosc", "average of sim and sim_nosc")]
    print("| arm | c_team | DFL sim | SC live | combined |\n|---|---|---|---|---|")
    chosen = {}
    for tag, label in arms:
        z = b1_compare.zone_score(tag)
        chosen[tag] = z["c_team"]
        print(f"| {label} | {z['c_team']} | {z['dfl_sim']:.3f} | {z['sc_live']:.3f} | "
              f"{z['combined']:.3f} |", flush=True)
    print()
    for tag, label in arms:
        print(f"## {label}\n")
        out = subprocess.run([sys.executable, os.path.join(HERE, "b1_durations.py"), tag,
                              f"{chosen[tag]},6,6"], capture_output=True, text=True).stdout
        print(out, flush=True)


if __name__ == "__main__":
    main()
