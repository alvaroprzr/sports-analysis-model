"""1c: ceilings of E7 with the control reading, from its saved development predictions (no
training). Each oracle replaces part of the probabilities with the truth and re-decodes:

    team    P(home | alive) perfect on frames that are truly alive
    short4  both probabilities perfect on the frames of true possessions shorter than 4 s
    short2  the same below 2 s
    dead    P(alive) perfect everywhere

Scored at E7's zone-score cost and with the cost re-chosen by the zone score. DFL sets use the
control reading; SkillCorner its own labels (team oracle on live frames; it has no dead-ball
labels to make the dead oracle on).

    python scripts/control_1c.py [TAG] > experiments/control_1c.md     (default ens_e7_ctl)
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_compare  # noqa: E402
import b1_dev  # noqa: E402
from poss import dataset, evaluate  # noqa: E402
from poss.evaluate import pm  # noqa: E402

HI, LO = 0.999, 0.001


def short_mask(gt, near, fps, limit_s):
    """Rows (10 Hz) that fall inside a true possession shorter than limit_s."""
    mask = np.zeros(len(gt), bool)
    for a, b, _ in pm.possession_spells(gt, pm.valid_runs(gt)):
        if (b - a) / fps < limit_s:
            mask[a:b] = True
    return mask[near]


def oracle(pred, kind):
    out = {}
    for k, r in pred.items():
        r = dict(r)
        g = r["gt"][r["near"]]
        pa, ph = r["p_alive"].copy(), r["p_home"].copy()
        known = g < dataset.X
        if kind == "team":
            live = g < dataset.D
            ph[live] = np.where(g[live] == dataset.H, HI, LO)
        elif kind in ("short4", "short2"):
            fps = 10 if k.startswith("sc_") else pm.FPS  # SkillCorner tracking is 10 fps
            sm = short_mask(r["gt"], r["near"], fps, 4.0 if kind == "short4" else 2.0) & known
            ph[sm & (g < dataset.D)] = np.where(g[sm & (g < dataset.D)] == dataset.H, HI, LO)
            if k.startswith("dfl"):
                pa[sm] = np.where(g[sm] != dataset.D, HI, LO)
        elif kind == "dead" and k.startswith("dfl"):
            pa[known] = np.where(g[known] != dataset.D, HI, LO)
        r["p_alive"], r["p_home"] = pa, ph
        out[k] = r
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "ens_e7_ctl"
    base = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))
    z0 = b1_compare.zone_score(tag)
    c0 = (z0["c_team"], b1_compare.ZONE_C_STOP, b1_compare.ZONE_C_STOP)
    print(f"Ceilings of {tag} (DFL: control reading). Base cost c_team {z0['c_team']}.\n")
    print("| oracle | DFL sim (base cost) | SC live (base cost) | combined (base cost) | "
          "c_team re-chosen | combined (re-chosen) |\n|---|---|---|---|---|---|")
    for kind in ("none", "team", "short4", "short2", "dead"):
        t = tag if kind == "none" else f"{tag}_or_{kind}"
        if kind != "none":
            pickle.dump(oracle(base, kind), open(os.path.join(b1_dev.CACHE, f"pred_{t}.pkl"), "wb"))
        s = b1_compare.per_match(t, c0)
        d = np.mean([v for k, v in s.items() if k.startswith("dfl_sim")])
        sc = np.mean([v for k, v in s.items() if k.startswith("sc_")])
        z = b1_compare.zone_score(t)
        print(f"| {kind} | {d:.3f} | {sc:.3f} | {evaluate.combined(d, sc):.3f} | {z['c_team']} | "
              f"{z['combined']:.3f} |", flush=True)


if __name__ == "__main__":
    main()
