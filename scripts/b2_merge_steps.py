"""Merge the best-step files (for --refit-from) and the predictions of validated B2 runs on
disjoint folds into one tag.

    python scripts/b2_merge_steps.py OUT_TAG TAG [TAG ...]     e.g.  b2cand b2e0 b2e0_f234
"""

import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402


def main():
    out, tags = sys.argv[1], sys.argv[2:]
    merged = {}
    for t in tags:
        for fold, steps in json.load(open(os.path.join(b1_dev.CACHE, f"steps_{t}.json"))).items():
            if fold in merged:
                raise ValueError(f"fold {fold} appears twice")
            merged[fold] = steps
    json.dump(merged, open(os.path.join(b1_dev.CACHE, f"steps_{out}.json"), "w"))
    pred = {}
    for t in tags:
        pred.update(pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{t}.pkl"), "rb")))
    pickle.dump(pred, open(os.path.join(b1_dev.CACHE, f"pred_{out}.pkl"), "wb"))
    print(out, dict(sorted(merged.items())))


if __name__ == "__main__":
    main()
