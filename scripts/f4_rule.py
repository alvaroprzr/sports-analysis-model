"""F4 adoption rule (confirmed 05/11), decoder cost chosen excluding the evaluated fold:
adopt a variant if the combined score, averaged over the two readings of the DFL labels, improves
by more than 2 paired SE, and no source (DFL sim, SkillCorner live) gets worse by more than 1 paired
SE in either reading.

    python scripts/f4_rule.py > experiments/f4_rule.md
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import f4_chain  # noqa: E402
import f4_decoder  # noqa: E402
import xfold  # noqa: E402
from poss import evaluate  # noqa: E402

W = evaluate.SC_WEIGHT


def per_set(tag, reading, kind=None):
    t = tag if reading == "flag" else xfold.with_control(tag)
    if kind is None:
        return xfold.scores(t)[0]
    if kind == "dmin1":
        return xfold.scores(t, f4_decoder.grid(t, lambda c: f4_decoder.dmin(c, 25)))[0]
    pred = f4_chain.load(t)
    pchange = {k: r["p_change"] for k, r in f4_chain.load("f4i_full_b2e5").items()}
    return f4_chain.xscores(f4_chain.grid(pred, kind, pchange))[0]


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "f4i"  # "f4x": the models of the external simulator
    cases = ([("geo", "f4_geo", None, "f4_arith", None), ("temp", "f4_temp", None, "f4_arith", None),
              ("dmin1", "f4_arith", "dmin1", "f4_arith", None)] if pre == "f4i" else [])
    cases += [("win", f"{pre}_win", None, f"{pre}_full", None),
              ("swap", f"{pre}_swap", None, f"{pre}_full", None),
              ("mindur", f"{pre}_full", "mindur", f"{pre}_full", None)]
    if pre == "f4i":
        cases.append(("varcost", "f4i_full", "varcost", "f4i_full", None))
    print("| variant | combined, mean of readings: diff (SE) | DFL flag | DFL control | SC (flag cost) "
          "| SC (control cost) | adopt |\n|---|---|---|---|---|---|---|")
    for name, tag, kind, ref, rkind in cases:
        d = {}
        for reading in ("flag", "control"):
            a, b = per_set(tag, reading, kind), per_set(ref, reading, rkind)
            d[reading] = {k: a[k] - b[k] for k in a}
        keys = list(d["flag"])
        dd = np.array([(d["flag"][k] + d["control"][k]) / 2 for k in keys if k.startswith("dfl_sim")])
        ds = np.array([(d["flag"][k] + d["control"][k]) / 2 for k in keys if k.startswith("sc_")])
        diff = (dd.mean() + W * ds.mean()) / (1 + W)
        s = np.sqrt(se(dd) ** 2 + (W * se(ds)) ** 2) / (1 + W)
        cells, ok = [], diff > 2 * s
        for reading in ("flag", "control"):
            pass
        for reading, prefix in (("flag", "dfl_sim"), ("control", "dfl_sim"), ("flag", "sc_"),
                                ("control", "sc_")):
            x = np.array([v for k, v in d[reading].items() if k.startswith(prefix)])
            cells.append(f"{x.mean():+.4f} ({se(x):.4f})")
            ok &= x.mean() >= -se(x)
        print(f"| {name} | {diff:+.4f} ({s:.4f}) | " + " | ".join(cells) + f" | {'YES' if ok else 'no'} |",
              flush=True)


if __name__ == "__main__":
    main()
