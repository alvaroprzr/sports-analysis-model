"""POST HOC, exploratory, after the rule gave its answer: where does the SkillCorner-live loss of arm c
come from, the decoder costs or the stoppages themselves? Nothing is chosen with this."""
import sys

import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alive_select as A  # noqa: E402
from poss import evaluate  # noqa: E402

sets, _ = A.load_sets()
dk = [k for k in sets if k.startswith("dfl")]
sk = [k for k in sets if k.startswith("sc_")]


def run(arm, c, cs):
    out = {}
    for k, s in sets.items():
        p = A.native(k, s, arm, c, cs, exact=True)
        if k.startswith("dfl"):
            f, ct = A.main_scores(k, s, p)
            out[k] = dict(flag=f, control=ct)
        else:
            out[k] = dict(m1=A.main_scores(k, s, p)[0],
                          live=evaluate.live_scores(s["gt"], p)["spellF1_live"],
                          raw=evaluate.full_scores(s["gt"], p)["spellF1"])
    return out


ref = run("frozen", 3, 8.0)
f = lambda d: f"{d[0]:+.4f} ({d[1]:.4f})"
print("| P(alive), decoder | SC metric 1 | SC live | DFL flag | DFL control | SC raw |")
print("|---|---|---|---|---|---|")
for arm, c, cs in (("c", 3, 8.0), ("c", 3, 6.0), ("c", 3, 12.0), ("c", 2, 6.0), ("a", 3, 8.0), ("d", 3, 8.0),
                   ("frozen", 2, 6.0)):
    r = run(arm, c, cs)
    print(f"| {arm} ({c}, {cs:g}) | {f(A.paired(r, ref, sk, 'm1'))} | {f(A.paired(r, ref, sk, 'live'))} | "
          f"{f(A.paired(r, ref, dk, 'flag'))} | {f(A.paired(r, ref, dk, 'control'))} | {f(A.paired(r, ref, sk, 'raw'))} |",
          flush=True)
