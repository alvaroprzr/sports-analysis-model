"""Adoption rule for the B1 reserve, fixed before the search results were read: a configuration
replaces the reference only if its combined score beats the reference's by more than 2 paired
standard errors and neither source gets worse. Each configuration is scored at its own
zone-selected team-switch cost (b1_compare.zone_score).

Paired SE of the combined difference, from per-match differences in each source:
    SE = sqrt(SE_DFL^2 + (w * SE_SC)^2) / (1 + w),   w = evaluate.SC_WEIGHT

    python scripts/b1_adopt.py N_CONFIGS > experiments/b1_adopt.md
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_compare  # noqa: E402
from poss import evaluate  # noqa: E402

# Params
N_SE = 2.0


def per_source(tag):
    z = b1_compare.zone_score(tag)
    s = b1_compare.per_match(tag, (z["c_team"], b1_compare.ZONE_C_STOP, b1_compare.ZONE_C_STOP))
    d = {k: v for k, v in s.items() if k.startswith("dfl_sim")}
    c = {k: v for k, v in s.items() if k.startswith("sc_")}
    return z, d, c


def main():
    n = int(sys.argv[1])
    w = evaluate.SC_WEIGHT
    zr, dr, cr = per_source("search0")
    print(f"Reference (config 0): c_team {zr['c_team']}, combined {zr['combined']:.3f}\n")
    print("| # | c_team | combined | diff | paired SE | DFL diff | SC diff | adopt |")
    print("|---|---|---|---|---|---|---|---|")
    adopted = []
    for j in range(1, n):
        z, d, c = per_source(f"search{j}")
        dd = np.array([d[k] - dr[k] for k in dr])
        dc = np.array([c[k] - cr[k] for k in cr])
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
        diff = z["combined"] - zr["combined"]
        sec = np.sqrt(se(dd) ** 2 + (w * se(dc)) ** 2) / (1 + w)
        ok = diff > N_SE * sec and dd.mean() >= 0 and dc.mean() >= 0
        if ok:
            adopted.append((diff, j))
        print(f"| {j} | {z['c_team']} | {z['combined']:.3f} | {diff:+.3f} | {sec:.3f} | "
              f"{dd.mean():+.3f} | {dc.mean():+.3f} | {'yes' if ok else 'no'} |", flush=True)
    print()
    if adopted:
        print(f"Adopt configuration {max(adopted)[1]} (largest qualifying gain).")
    else:
        print("No configuration qualifies: the reserve stays as it is. B1 is on a plateau over this "
              "space; further gains have to come from B2.")


if __name__ == "__main__":
    main()
