"""Screening of one factor (2 folds) against its reference on the same held-out sets, with the
decoder cost chosen excluding the evaluated fold (scripts/xfold.py), on both readings of the DFL
labels. Screening rule (approved 05/11): passes if the combined score improves and no source
(DFL sim with the control reading, DFL sim with the flag, SkillCorner live) drops by more than
1 paired SE. The strict rule is applied later, in the single final retraining on 5 folds.

    python scripts/compare_factor.py NEW_TAG REF_TAG > experiments/factor_<name>.md
"""

import os
import pickle
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import xfold  # noqa: E402


def restrict(ref, keys):
    out = f"{ref}_on_{len(keys)}"
    p = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{ref}.pkl"), "rb"))
    pickle.dump({k: p[k] for k in keys}, open(os.path.join(b1_dev.CACHE, f"pred_{out}.pkl"), "wb"))
    for suffix in ("_ctl",):
        path = os.path.join(b1_dev.CACHE, f"pred_{out}{suffix}.pkl")
        if os.path.exists(path):
            os.remove(path)
    return out


def main():
    new, ref = sys.argv[1], sys.argv[2]
    keys = list(pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{new}.pkl"), "rb")))
    ref = restrict(ref, keys)
    res = {}
    for tag in (ref, new):
        for reading, t in (("control", xfold.with_control(tag)), ("flag", tag)):
            res[(tag, reading)] = xfold.scores(t)
    print(f"Held-out sets: {len(keys)}. Cost chosen excluding the evaluated fold.\n")
    print("| model | reading | c_team per fold | DFL sim | SC live | combined |\n|---|---|---|---|---|---|")
    for (tag, reading), (s, ch) in res.items():
        d, sc, cmb = xfold.summary(s)
        print(f"| {tag} | {reading} | {list(ch.values())} | {d:.3f} | {sc:.3f} | {cmb:.3f} |")
    diff, se = xfold.combined_paired(res[(new, "control")][0], res[(ref, "control")][0])
    checks = []
    for name, reading, prefix in (("DFL sim, control", "control", "dfl_sim"),
                                  ("DFL sim, flag", "flag", "dfl_sim"),
                                  ("SkillCorner live", "control", "sc_")):
        m, s, _ = xfold.paired(res[(new, reading)][0], res[(ref, reading)][0], prefix)
        checks.append(m >= -s)
        print(f"- {name}: {m:+.3f} (paired SE {s:.3f})")
    ok = diff > 0 and all(checks)
    print(f"- combined, control: {diff:+.3f} (paired SE {se:.3f})\n\nPasses the screening: "
          f"{'YES' if ok else 'no'}\n")
    for tag, reading in ((ref, "control"), (new, "control")):
        t = xfold.with_control(tag)
        c = max(set(res[(tag, reading)][1].values()), key=list(res[(tag, reading)][1].values()).count)
        print(f"## {tag}, control reading, c_team {c}\n")
        print(subprocess.run([sys.executable, os.path.join(HERE, "b1_durations.py"), t, f"{c},6,6"],
                             capture_output=True, text=True).stdout, flush=True)


if __name__ == "__main__":
    main()
