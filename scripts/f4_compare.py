"""Compare decoder- or inference-level variants against a reference on the same held-out sets,
decoder cost chosen excluding the evaluated fold, both readings of the DFL labels.

    python scripts/f4_compare.py REF VARIANT [VARIANT ...] > experiments/f4_<name>.md
"""

import sys

import xfold


def main():
    ref, variants = sys.argv[1], sys.argv[2:]
    rows = {}
    for tag in [ref] + variants:
        for reading, t in (("flag", tag), ("control", xfold.with_control(tag))):
            rows[(tag, reading)] = xfold.scores(t)
    print("| variant | reading | c_team per fold | DFL sim | SC live | combined |\n|---|---|---|---|---|---|")
    for (tag, reading), (s, ch) in rows.items():
        d, sc, cmb = xfold.summary(s)
        print(f"| {tag} | {reading} | {list(ch.values())} | {d:.3f} | {sc:.3f} | {cmb:.3f} |")
    print(f"\nPaired differences against {ref} (combined: diff (SE); DFL sim; SC live):\n")
    for tag in variants:
        cells = []
        for reading in ("flag", "control"):
            a, b = rows[(tag, reading)][0], rows[(ref, reading)][0]
            diff, se = xfold.combined_paired(a, b)
            md, sd, _ = xfold.paired(a, b, "dfl_sim")
            ms, ss, _ = xfold.paired(a, b, "sc_")
            cells.append(f"{reading}: {diff:+.4f} ({se:.4f}), DFL {md:+.4f} ({sd:.4f}), "
                         f"SC {ms:+.4f} ({ss:.4f})")
        print(f"- {tag}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
