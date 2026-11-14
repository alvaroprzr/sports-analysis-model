"""1d: E0 trained on the control reading of the DFL labels against E0 trained on the flag, on the
5 development folds, with the decoder cost chosen excluding the evaluated fold (scripts/xfold.py).

Adoption rule (fixed 05/11 before any result): adopt if all three hold:
  a. combined score with the control reading better by more than 2 paired SE;
  b. the mean of the two readings (control and flag) on simulated DFL does not go down;
  c. SkillCorner live not worse by more than 1 paired SE.

    python scripts/compare_1d.py NEW_TAG [REF_TAG] > experiments/compare_1d.md
"""

import sys

import xfold

N_SE_A, N_SE_C = 2.0, 1.0


def main():
    new = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else "b2e0_5f"
    res = {}
    for tag in (ref, new):
        for reading, t in (("flag", tag), ("control", xfold.with_control(tag))):
            res[(tag, reading)] = xfold.scores(t)
    print(f"Decoder cost chosen excluding the evaluated fold. New: {new}; reference: {ref}.\n")
    print("| model | reading | c_team per fold | DFL sim | SC live | combined |\n|---|---|---|---|---|---|")
    for (tag, reading), (s, ch) in res.items():
        d, sc, cmb = xfold.summary(s)
        print(f"| {tag} | {reading} | {list(ch.values())} | {d:.3f} | {sc:.3f} | {cmb:.3f} |")
    a_new, a_ref = res[(new, "control")][0], res[(ref, "control")][0]
    diff_a, se_a = xfold.combined_paired(a_new, a_ref)
    f_new, f_ref = res[(new, "flag")][0], res[(ref, "flag")][0]
    mean_new = (xfold.summary(a_new)[0] + xfold.summary(f_new)[0]) / 2
    mean_ref = (xfold.summary(a_ref)[0] + xfold.summary(f_ref)[0]) / 2
    md_sc, se_sc, _ = xfold.paired(a_new, a_ref, "sc_")
    ok_a = diff_a > N_SE_A * se_a
    ok_b = mean_new >= mean_ref
    ok_c = md_sc >= -N_SE_C * se_sc
    print(f"\na. combined (control): {diff_a:+.3f}, paired SE {se_a:.3f} -> {'yes' if ok_a else 'no'}")
    print(f"b. DFL sim, mean of the two readings: {mean_new:.3f} vs {mean_ref:.3f} -> "
          f"{'yes' if ok_b else 'no'}")
    print(f"c. SkillCorner live: {md_sc:+.3f}, paired SE {se_sc:.3f} -> {'yes' if ok_c else 'no'}")
    print(f"\nAdopt: {'YES' if ok_a and ok_b and ok_c else 'no'}")


if __name__ == "__main__":
    main()
