"""E7: combination c (E0 + E1 + E2 + E5, equal-weight probability average) on the 5 development
folds, against E0 alone with the same early-stopped training (b2e0_5f). Same adoption rule as
scripts/b2_ensembles.py: replace E0 only if the combined zone score is better by more than 2 paired
standard errors and no source gets worse.

    python scripts/b2_e7.py > experiments/b2_e7.md
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_adopt  # noqa: E402
import b2_ensembles as ens  # noqa: E402
from poss import evaluate  # noqa: E402

REF = "b2e0_5f"
TAGS = ["b2e0_5f", "b2e1_5f", "b2e2_5f", "b2e5_5f"]


def main():
    ens.save("ens_e7", ens.average(TAGS))
    w = evaluate.SC_WEIGHT
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    zr, dr, cr = b1_adopt.per_source(REF)
    print("| model | c_team | DFL sim | SC live | combined | diff | paired SE | adopt |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| E0 ({REF}) | {zr['c_team']} | {zr['dfl_sim']:.3f} | {zr['sc_live']:.3f} | "
          f"{zr['combined']:.3f} | | | |")
    for tag in TAGS[1:] + ["ens_e7"]:
        z, d, c = b1_adopt.per_source(tag)
        dd = np.array([d[k] - dr[k] for k in dr])
        dc = np.array([c[k] - cr[k] for k in cr])
        diff = z["combined"] - zr["combined"]
        sec = np.sqrt(se(dd) ** 2 + (w * se(dc)) ** 2) / (1 + w)
        ok = diff > ens.N_SE * sec and dd.mean() >= 0 and dc.mean() >= 0
        adopt = ("yes" if ok else "no") if tag == "ens_e7" else ""
        print(f"| {tag} | {z['c_team']} | {z['dfl_sim']:.3f} | {z['sc_live']:.3f} | "
              f"{z['combined']:.3f} | {diff:+.3f} | {sec:.3f} | {adopt} |", flush=True)


if __name__ == "__main__":
    main()
