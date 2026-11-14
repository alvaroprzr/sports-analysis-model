"""1b: the control reading of the DFL labels (src/poss/control.py), without training anything.

1. Builds the control labels of the 5 development DFL matches (data/cache/control/<mid>.npz),
   with the event-clock shift per half and how each would-be change of possession was decided.
2. Possessions per match and by duration with the flag and the control reading (official spell
   definition of evaluation/possession_metrics.py).
3. Re-scores the saved development predictions of B1, E0 and E7 with the control reading (DFL
   sets only; SkillCorner unchanged) and reports the decoder cost the zone score picks with each.

    python scripts/control_1b.py > experiments/control_1b.md
    python scripts/b1_durations.py <tag>_ctl c,6,6      # duration tables with the control reading
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import b1_compare  # noqa: E402
import b1_dev  # noqa: E402
from poss.evaluate import pm  # noqa: E402
from poss import control, dataset, evaluate  # noqa: E402

TAGS = {"B1": "sim", "E0": "b2e0_5f", "E7": "ens_e7"}
OUT = os.path.join(b1_dev.CACHE, "control")
BINS = evaluate.DURATION_BINS_S


def build(mid):
    path = os.path.join(OUT, f"{mid}.npz")
    m = b1_dev.load("dfl", mid)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return m, z["team"], z["info"].item()
    events, kickoff = control.load_events(mid)
    shifts = control.fit_shift(m, events, kickoff)
    team, counts = control.relabel(m, events, kickoff, shifts)
    info = dict(shifts=shifts, counts=counts, n_events=len(events))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, team=team, info=np.array(info, dtype=object))
    return m, team, info


def spells(codes, fps):
    s = pm.possession_spells(codes, pm.valid_runs(codes))
    return np.array([(b - a) / fps for a, b, _ in s])


def main():
    dev = b1_dev.SPLIT["dfl"]["dev"]
    gt_ctl = {}
    print("## Control labels\n")
    print("| match | events | shift h1 (s, y sign, median m) | shift h2 | live runs | would-be "
          "changes | kept: event | kept: >= 3 s | kept: passes | reverted |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    dur = {"flag": [], "control": []}
    per_match = []
    for mid in dev:
        m, team, info = build(mid)
        c = info["counts"]
        sh = info["shifts"]
        fmt = lambda h: f"{sh[h][0]:+.2f}, {sh[h][1]:+d}, {sh[h][2]:.2f}" if h in sh else "-"
        print(f"| {mid} | {info['n_events']} | {fmt(1)} | {fmt(2)} | {c['runs']} | {c['changes']} "
              f"| {c['by_event']} | {c['by_duration']} | {c['by_passes']} | {c['reverted']} |")
        g_flag = dataset.codes(m.alive, m.team, m.period)
        g_ctl = dataset.codes(m.alive, team, m.period)
        gt_ctl[mid] = g_ctl
        d_f, d_c = spells(g_flag, m.fps), spells(g_ctl, m.fps)
        dur["flag"].append(d_f)
        dur["control"].append(d_c)
        live_min = (m.alive == 1).sum() / m.fps / 60
        per_match.append((mid, len(d_f), len(d_c), live_min, np.median(d_f), np.median(d_c)))
    print("\n## Possessions per match\n")
    print("| match | flag | control | per live minute (flag / control) | median s (flag / control) |")
    print("|---|---|---|---|---|")
    for mid, nf, nc, lm, mf, mc in per_match:
        print(f"| {mid} | {nf} | {nc} | {nf / lm:.2f} / {nc / lm:.2f} | {mf:.1f} / {mc:.1f} |")
    print("\n## Possessions by duration (5 development matches)\n")
    print("| duration | flag | control |\n|---|---|---|")
    for i in range(len(BINS) - 1):
        cnt = [int(((np.concatenate(dur[k]) >= BINS[i]) & (np.concatenate(dur[k]) < BINS[i + 1])).sum())
               for k in ("flag", "control")]
        hi = f"{BINS[i + 1]:g}" if np.isfinite(BINS[i + 1]) else "inf"
        print(f"| {BINS[i]:g}-{hi} s | {cnt[0]} | {cnt[1]} |")

    print("\n## Re-scored development predictions (zone score with each reading)\n")
    print("| model | reading | c_team | DFL sim | SC live | combined |\n|---|---|---|---|---|---|")
    for name, tag in TAGS.items():
        pred = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_{tag}.pkl"), "rb"))
        for k, r in pred.items():
            if k.startswith("dfl"):
                mid = k.split("_")[-1]
                assert len(r["gt"]) == len(gt_ctl[mid]), k
                r["gt"] = gt_ctl[mid]
        pickle.dump(pred, open(os.path.join(b1_dev.CACHE, f"pred_{tag}_ctl.pkl"), "wb"))
        for reading, t in (("flag", tag), ("control", tag + "_ctl")):
            z = b1_compare.zone_score(t)
            print(f"| {name} | {reading} | {z['c_team']} | {z['dfl_sim']:.3f} | "
                  f"{z['sc_live']:.3f} | {z['combined']:.3f} |", flush=True)


if __name__ == "__main__":
    main()
