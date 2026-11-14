"""Paired comparison of B1 arms at fixed decoder costs: per-match score differences (mean +- SE)
on each held-out set, so the simulator decision rests on the downstream effect.

    python scripts/b1_compare.py COSTS ARM_A ARM_B     e.g.  3,6,6 sim clean
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
from poss import decode, evaluate  # noqa: E402


def per_match(arm, costs):
    pred = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{arm}.pkl"), "rb"))
    out = {}
    for name, r in pred.items():
        c = decode.to_native(decode.decode(r["p_alive"], r["p_home"], r["period"], costs),
                             r["near"], r["n"])
        if name.startswith("dfl"):
            out[name] = evaluate.full_scores(r["gt"], c)["spellF1"]
        else:
            out[name] = evaluate.live_scores(r["gt"], c)["spellF1_live"]
    return out


def main():
    costs = tuple(float(x) for x in sys.argv[1].split(","))
    a, b = per_match(sys.argv[2], costs), per_match(sys.argv[3], costs)
    print(f"costs {costs}: {sys.argv[2]} minus {sys.argv[3]}\n")
    print("| held-out set | n | mean A | mean B | A - B | SE | wins A |\n|---|---|---|---|---|---|---|")
    for prefix, label in (("dfl_sim", "DFL simulated"), ("dfl_clean", "DFL clean"),
                          ("sc_", "SkillCorner live")):
        keys = [k for k in a if k.startswith(prefix)]
        d = np.array([a[k] - b[k] for k in keys])
        print(f"| {label} | {len(keys)} | {np.mean([a[k] for k in keys]):.3f} | "
              f"{np.mean([b[k] for k in keys]):.3f} | {d.mean():+.3f} | "
              f"{d.std(ddof=1) / np.sqrt(len(d)):.3f} | {(d > 0).sum()}/{len(d)} |")


if __name__ == "__main__":
    main()


# Params
ZONE_C_TEAM = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24)
ZONE_C_STOP = 6.0  # the stoppage costs had no effect between 2 and 12 (experiments/b1_compare.md)


def zone_score(tag):
    """Selection score that does not follow either source's granularity: over the team-switch
    costs, take the one where the worse of the two sources is closest to its own best, and report
    the combined score there (with each source's mean and the chosen cost)."""
    rows = []
    for c in ZONE_C_TEAM:
        s = per_match(tag, (c, ZONE_C_STOP, ZONE_C_STOP))
        d = np.mean([v for k, v in s.items() if k.startswith("dfl_sim")])
        sc = [v for k, v in s.items() if k.startswith("sc_")]
        rows.append((c, d, np.mean(sc) if sc else np.nan))
    best_d = max(r[1] for r in rows)
    best_s = max(r[2] for r in rows)
    gap = [max(best_d - d, (best_s - s) if np.isfinite(s) else 0.0) for _, d, s in rows]
    c, d, s = rows[int(np.argmin(gap))]
    return {"c_team": c, "dfl_sim": d, "sc_live": s,
            "combined": evaluate.combined(d, s) if np.isfinite(s) else d}
