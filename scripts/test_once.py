"""The reserved test (configs/splits.json: 2 DFL, 3 SkillCorner matches never used to train, choose
or tune), queried once, for information only: nothing is changed after it.

    POSS_SIM=ext_dr python scripts/test_once.py predict
        the fold models of every E7 member (data/cache/b1/models/<member>_f<i>_f<i>.pt) predict the
        test sets (simulated DFL at the held-out degradation, SkillCorner as is); per member the
        mean over its five fold models, plain and with the teams exchanged; nothing is scored.
    POSS_SIM=ext_dr python scripts/test_once.py score CONFIG > experiments/test_once.md
        the decoder of CONFIG (the submission's: costs, team swap, minimum duration, F5,
        boundary shifts) on the equal-weight member mean; spellF1 per set with the flag and
        the control reading of the DFL labels, SkillCorner live. Refuses to run a second time.
"""

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
from make_submission_b2ens import ROUND, load_model  # noqa: E402
from poss import b2, ballout, control, dataset, decode, decode2, evaluate, shift  # noqa: E402

MEMBERS = ("b2e0d", "b2e1d", "b2e2d", "b2e5d")
PRED = os.path.join(b1_dev.CACHE, "pred_test_e7d.pkl")
DONE = os.path.join(b1_dev.CACHE, "test_once.done")


def test_sets():
    sets = [(f"dfl_sim_{m}", "dfl", m) for m in b1_dev.SPLIT["dfl"]["test"]]
    return sets + [(f"sc_{m}", "skillcorner", m) for m in b1_dev.SPLIT["skillcorner"]["test"]]


def make_copy(src, mid, swap):
    return (b2_dev.dfl_copy(mid, b1_dev.TEST_SEED, swap) if src == "dfl" else b2_dev.sc_copy(mid, swap))


def predict():
    b2.set_threads(int(os.environ.get("THREADS", "4")))
    out = {}
    for key, src, mid in test_sets():
        m = b1_dev.load(src, mid)
        _, _, _, period, near = (b1_dev.dfl_rows(mid, b1_dev.TEST_SEED, False) if src == "dfl"
                                 else b1_dev.sc_rows(mid, False))
        c0, c1 = make_copy(src, mid, False), make_copy(src, mid, True)
        assert len(c0["y"]) == len(near) and np.array_equal(c0["period"], period)
        rec = dict(period=period, near=near, n=m.n_frames, gt=dataset.codes(m.alive, m.team, m.period),
                   ball=c0["ball"], players=c0["players"], half=float(m.pitch[0]) / 2, members={})
        if src == "dfl":
            events, kickoff = control.load_events(mid)
            team, _ = control.relabel(m, events, kickoff, control.fit_shift(m, events, kickoff))
            rec["gt_control"] = dataset.codes(m.alive, team, m.period)
        for tag in MEMBERS:
            f = {"pa": [], "ph": [], "pa_sw": [], "ph_sw": []}
            for i in range(b1_dev.N_FOLDS):
                model, stats = load_model(os.path.join(b2_dev.MODELS, f"{tag}_f{i}_f{i}.pt"))
                pa, ph = b2_dev.predict(model, c0, stats)
                pa2, ph2 = b2_dev.predict(model, c1, stats)
                f["pa"].append(pa), f["ph"].append(ph), f["pa_sw"].append(pa2), f["ph_sw"].append(ph2)
            rec["members"][tag] = {k: np.mean(v, 0) for k, v in f.items()}
        out[key] = rec
        print(key, "predicted", flush=True)
    pickle.dump(out, open(PRED, "wb"))


def score(config):
    if os.path.exists(DONE):
        sys.exit("the reserved test was already queried once (" + DONE + ")")
    conf = json.load(open(config))
    pred = pickle.load(open(PRED, "rb"))
    costs = tuple(conf["costs"])
    swap = conf.get("team_swap", False)
    k_rows = int(conf.get("min_possession_rows", 1))
    bo = conf.get("ball_out")
    rows = []
    for key, r in pred.items():
        pas, phs = [], []
        for tag in MEMBERS:
            e = r["members"][tag]
            pa, ph = e["pa"], e["ph"]
            if swap:
                pa, ph = (pa + e["pa_sw"]) / 2, (ph + 1 - e["ph_sw"]) / 2
            pas.append(pa), phs.append(ph)
        pa = np.mean(pas, 0)
        if bo:
            pa = ballout.apply(pa, r["ball"], bo["min_out_m"], bo["min_s"])
        pa, ph = np.round(pa, ROUND), np.round(np.mean(phs, 0), ROUND)
        c10 = (decode.decode(pa, ph, r["period"], costs) if k_rows == 1
               else decode2.decode(pa, ph, r["period"], costs, k=k_rows))
        if conf.get("boundary_shift_s"):
            c10 = shift.apply(c10, r["period"], conf["boundary_shift_s"])
        nat = decode.to_native(c10, r["near"], r["n"])
        if key.startswith("dfl"):
            rows.append((key, evaluate.full_scores(r["gt"], nat)["spellF1"],
                         evaluate.full_scores(r["gt_control"], nat)["spellF1"]))
        else:
            rows.append((key, evaluate.live_scores(r["gt"], nat)["spellF1_live"], None))
    open(DONE, "w").write(config + "\n")
    print(f"Reserved test, queried once with the decoder of {os.path.basename(config)} and the fold "
          f"models of {', '.join(MEMBERS)} (mean over the five folds per member). For information "
          f"only.\n")
    print("| set | spellF1 (flag / live) | spellF1 (control reading) |\n|---|---|---|")
    for key, a, b in rows:
        print(f"| {key} | {a:.3f} | " + (f"{b:.3f}" if b is not None else "-") + " |")
    d = np.mean([a for k, a, _ in rows if k.startswith("dfl")])
    dc = np.mean([b for k, _, b in rows if k.startswith("dfl")])
    s = np.mean([a for k, a, _ in rows if k.startswith("sc_")])
    print(f"\nMeans: DFL sim {d:.3f} (control {dc:.3f}), SkillCorner live {s:.3f}; combined "
          f"{evaluate.combined(d, s):.3f} (control {evaluate.combined(dc, s):.3f}).")
    print("\nNote: the two DFL test matches were seen in earlier sessions of other pipelines "
          "(configs/splits.json); the SkillCorner test is the clean one.")


if __name__ == "__main__":
    predict() if sys.argv[1] == "predict" else score(sys.argv[2])
