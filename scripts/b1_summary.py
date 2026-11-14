"""Summarise a b1_dev.py decode table by the team-switch cost (the granularity lever): range of
each source's score over the other two costs, and the number of predicted DFL possessions.

    python scripts/b1_summary.py experiments/b1_decode_<arm>.md
"""

import collections
import re
import sys

import numpy as np


def main(path):
    by = collections.defaultdict(list)
    for line in open(path):
        if not re.match(r"\| \d", line):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        by[float(c[0])].append((float(c[3].split()[0]), float(c[5].split()[0]), float(c[7]),
                                int(c[8].split("/")[0])))
    best_d = max(max(v[0] for v in r) for r in by.values())
    best_s = max(max(v[1] for v in r) for r in by.values())
    print("| c_team | DFL sim | gap to DFL best | SC live | gap to SC best | combined | DFL poss |")
    print("|---|---|---|---|---|---|---|")
    for ct in sorted(by):
        v = np.array(by[ct])
        print(f"| {ct:g} | {v[:, 0].min():.3f}-{v[:, 0].max():.3f} | {best_d - v[:, 0].max():.3f} | "
              f"{v[:, 1].min():.3f}-{v[:, 1].max():.3f} | {best_s - v[:, 1].max():.3f} | "
              f"{v[:, 2].max():.3f} | {int(v[:, 3].min())}-{int(v[:, 3].max())} |")


if __name__ == "__main__":
    main(sys.argv[1])
