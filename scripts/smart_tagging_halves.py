"""Alignment check of the second half (label-free on BAR-ATH, aggregates only, never used to choose):
per half, for the corners, throw-ins and goal kicks of Metrica's Smart Tagging outside the public
window, (a) the share with a predicted stoppage (D) within +-3 s of the tag's start and (b) for those
whose tag names a team, the share whose first predicted possession in the 10 s after the start is
that team. A second half clearly worse than the first would point to a frame-offset or side-change
error.

    python scripts/smart_tagging_halves.py CSV > experiments/smart_tagging_halves.md
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import load_smart_tagging  # noqa: E402
from games import GAME  # noqa: E402

TYPES = ("Corner Kick", "Throw-in", "Goal Kick")
TEAM = {"barcelona": "H", "atletic club": "A"}
PUBLIC = (11185, 18684)
FPS, NEAR_S, AFTER_S = 25, 3, 10


def main():
    lab = pd.read_csv(sys.argv[1])["label"].to_numpy()
    tags = [t for t in load_smart_tagging.load(GAME) if t["code"] == "SET PIECES"
            and t["tags"].get("Type") in TYPES and not (PUBLIC[0] <= t["start_frame"] <= PUBLIC[1])]
    print(f"File: {os.path.basename(sys.argv[1])}. Set pieces: {', '.join(TYPES)}; frames 11185-18684 excluded.\n")
    print("| half | set pieces | D within +-3 s | with a team tag | first possession = tagged team |")
    print("|---|---|---|---|---|")
    for half in ("1st Half", "2nd Half"):
        sel = [t for t in tags if t["tags"].get("Half") == half]
        near_d = [bool((lab[max(0, t["start_frame"] - NEAR_S * FPS):t["start_frame"] + NEAR_S * FPS] == "D").any())
                  for t in sel]
        teamed = [t for t in sel if str(t["tags"].get("Team", "")).lower() in TEAM]
        agree = 0
        for t in teamed:
            seg = lab[t["start_frame"]:t["start_frame"] + AFTER_S * FPS]
            first = next((x for x in seg if x in ("H", "A")), None)
            agree += int(first == TEAM[t["tags"]["Team"].lower()])
        print(f"| {half} | {len(sel)} | {sum(near_d)}/{len(sel)} ({np.mean(near_d):.2f}) | {len(teamed)} | "
              f"{agree}/{len(teamed)}" + (f" ({agree / len(teamed):.2f})" if teamed else "") + " |")
    for typ in TYPES:
        cells = []
        for half in ("1st Half", "2nd Half"):
            sel = [t for t in tags if t["tags"].get("Half") == half and t["tags"]["Type"] == typ]
            n = sum(bool((lab[max(0, t["start_frame"] - NEAR_S * FPS):t["start_frame"] + NEAR_S * FPS] == "D").any())
                    for t in sel)
            cells.append(f"{n}/{len(sel)}")
        print(f"\n- {typ}: D within +-3 s, 1st half {cells[0]}, 2nd half {cells[1]}", end="")
    print()


if __name__ == "__main__":
    main()
