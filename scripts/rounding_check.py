"""Effect of rounding both probabilities to 1e-4 before the decoder, on the development held-out
sets of the submitted models (B1 and the E7 combination), at their submission costs.

    python scripts/rounding_check.py > experiments/rounding_submitted.md
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_compare  # noqa: E402
import b2_ensembles as ens  # noqa: E402
from poss import decode  # noqa: E402

ROUND = 4
MODELS = {"B1 (sim)": ("sim", (2.5, 6, 6)), "E7 (ens_e7)": ("ens_e7", (3, 6, 6))}


def main():
    print("Rounding both probabilities to 1e-4 before decoding (development held-out sets)\n")
    print("| model | costs | sets | frames changed | max abs spellF1 change | mean change |")
    print("|---|---|---|---|---|---|")
    for name, (tag, costs) in MODELS.items():
        pred = ens.load(tag)
        rounded = {k: dict(r, p_alive=np.round(r["p_alive"], ROUND),
                           p_home=np.round(r["p_home"], ROUND)) for k, r in pred.items()}
        ens.save(tag + "_r4", rounded)
        changed = total = 0
        for k, r in pred.items():
            a = decode.decode(r["p_alive"], r["p_home"], r["period"], costs)
            b = decode.decode(rounded[k]["p_alive"], rounded[k]["p_home"], r["period"], costs)
            changed += int((a != b).sum())
            total += len(a)
        s0, s1 = b1_compare.per_match(tag, costs), b1_compare.per_match(tag + "_r4", costs)
        d = np.array([s1[k] - s0[k] for k in s0])
        print(f"| {name} | {costs} | {len(d)} | {changed} of {total} | {np.abs(d).max():.4f} | "
              f"{d.mean():+.5f} |", flush=True)


if __name__ == "__main__":
    main()
