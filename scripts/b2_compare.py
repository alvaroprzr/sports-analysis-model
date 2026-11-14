"""B2 variants against B1 on the same held-out sets (the folds the variants were trained on):
zone score per source and combined, and the duration tables, B1 restricted to the same matches.

    python scripts/b2_compare.py TAG [TAG ...] > experiments/b2_factors.md
"""

import os
import pickle
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_compare  # noqa: E402
import b1_dev  # noqa: E402

B1_TAG = "sim"


def restricted_b1(keys, tag):
    """B1's predictions on exactly the held-out sets of `keys`, saved as pred_<tag>.pkl."""
    p = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{B1_TAG}.pkl"), "rb"))
    pickle.dump({k: p[k] for k in keys}, open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "wb"))


def main():
    tags = sys.argv[1:]
    keys = list(pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tags[0]}.pkl"), "rb")))
    restricted_b1(keys, "b1_same")
    rows = [("b1_same", "B1 (same held-out sets)")] + [(t, t) for t in tags]
    print(f"Held-out sets: {len(keys)} ({sum(k.startswith('dfl_sim') for k in keys)} DFL matches)\n")
    print("| model | c_team | DFL sim | SC live | combined |\n|---|---|---|---|---|")
    chosen = {}
    for tag, label in rows:
        z = b1_compare.zone_score(tag)
        chosen[tag] = z["c_team"]
        print(f"| {label} | {z['c_team']} | {z['dfl_sim']:.3f} | {z['sc_live']:.3f} | "
              f"{z['combined']:.3f} |", flush=True)
    print()
    for tag, label in rows:
        print(f"## {label}\n")
        print(subprocess.run([sys.executable, os.path.join(HERE, "b1_durations.py"), tag,
                              f"{chosen[tag]},6,6"], capture_output=True, text=True).stdout,
              flush=True)


if __name__ == "__main__":
    main()
