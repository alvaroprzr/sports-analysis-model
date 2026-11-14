"""Diagnostic 12 (label-free, BAR-ATH): agreement of the submitted files with Metrica's automatic
Smart Tagging (FAQ.md: an automatic output, not ground truth). Only to detect gross failures (for
example teams swapped in a half), never to choose between variants. Frames 11185-18684
are excluded.

Team semantics, fixed before looking at any agreement: codes describing the team WITH the ball
(BUILD UP, PROGRESSION, BALL IN FINAL THIRD, CREATING CHANCES, ATTACKING TRANSITION, LONG BALL,
BALL IN THE BOX) are compared with the predicted team; codes describing the team WITHOUT it
(DEFENDING IN ... THIRD, DEFENSIVE TRANSITION) with the other team; PLAYERS IN THE BOX and GOALS
are reported as tagged, their meaning being unclear. Per phase: share of predicted live frames
(H / A) inside it that agree. Set pieces: the team of the first predicted possession starting in
the 10 s after the tag's start, against the tag's team (diagnostic 14).

    python scripts/smart_tagging_check.py LABEL=CSV [LABEL=CSV ...] > experiments/smart_tagging.md
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import load_smart_tagging  # noqa: E402
from games import GAME  # noqa: E402

WITH = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
        "ATTACKING TRANSITION", "LONG BALL", "BALL IN THE BOX"}
AGAINST = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
           "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"}
TEAM = {"barcelona": "H", "atletic club": "A"}  # DATA.md: home is Barcelona
PUBLIC = (11185, 18684)
SET_WINDOW_S = 10


def outside_public(a, b):
    """Frames of [a, b) outside the excluded range."""
    f = np.arange(a, b)
    return f[(f < PUBLIC[0]) | (f > PUBLIC[1])]


def main():
    tags = load_smart_tagging.load(GAME)
    files = dict(arg.split("=", 1) for arg in sys.argv[1:])
    labs = {n: pd.read_csv(p)["label"].to_numpy() for n, p in files.items()}
    print("## Phases: share of predicted live frames agreeing with the tagged team\n")
    print("| code | semantics | phases | " + " | ".join(labs) + " |\n|---|---|---|" + "---|" * len(labs))
    codes = sorted({t["code"] for t in tags if t["code"] != "SET PIECES"})
    for code in codes:
        sem = "with ball" if code in WITH else "without ball" if code in AGAINST else "as tagged"
        rows = [t for t in tags if t["code"] == code and TEAM.get(str(t["tags"].get("Team", "")).lower())]
        cells = []
        for n, lab in labs.items():
            agree = total = 0
            for t in rows:
                f = outside_public(t["start_frame"], t["end_frame"])
                seg = lab[f]
                live = seg[(seg == "H") | (seg == "A")]
                want = TEAM[t["tags"]["Team"].lower()]
                if code in AGAINST:
                    want = "A" if want == "H" else "H"
                agree += int((live == want).sum())
                total += len(live)
            cells.append(f"{agree / total:.2f}" if total else "-")
        print(f"| {code} | {sem} | {len(rows)} | " + " | ".join(cells) + " |")
    print("\n## Set pieces: team of the first predicted possession after the tag\n")
    print("| file | set pieces (frames 11185-18684 excluded) | agree | opposite | no possession in 10 s |")
    print("|---|---|---|---|---|")
    sp = [t for t in tags if t["code"] == "SET PIECES" and TEAM.get(str(t["tags"].get("Team", "")).lower())
          and not (PUBLIC[0] <= t["start_frame"] <= PUBLIC[1])]
    for n, lab in labs.items():
        agree = opp = none = 0
        for t in sp:
            seg = lab[t["start_frame"]:t["start_frame"] + SET_WINDOW_S * 25]
            first = next((x for x in seg if x in ("H", "A")), None)
            if first is None:
                none += 1
            elif first == TEAM[t["tags"]["Team"].lower()]:
                agree += 1
            else:
                opp += 1
        print(f"| {n} | {len(sp)} | {agree} | {opp} | {none} |")


if __name__ == "__main__":
    main()
