"""F4, part 2: inference-level variants of E7 on the development held-out sets, re-predicting with
the saved weights of each member and fold (data/cache/b1/models/<tag>_f<i>_f<i>.pt), so that every
variant and its baseline use the same weights:

    full   the current inference: the recurrent network reads each half in one pass
    win    overlapping 60 s windows (the training length), 30 s stride; each 5 Hz step takes the
           output of the window in which it is most central
    swap   mean of the prediction and that of the same input with the teams exchanged
           (P(alive) averaged; P(home) averaged with 1 - P(home) of the swapped input)
    E5's "possession changes here" output is also stored (p_change, full inference) for the
    variable switch cost.

Writes pred_f4i_<mode>_<member>.pkl and the equal-weight member means pred_f4i_<mode>.pkl.

    python scripts/f4_infer.py
"""

import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
from make_submission_b2ens import load_model  # noqa: E402
from poss import b2, dataset  # noqa: E402

MEMBERS = ["b2e0", "b2e1", "b2e2", "b2e5"]
# python scripts/f4_infer.py [SUFFIX [OUT]]: members b2e0<SUFFIX>... (e.g. "x" for the models of the
# external simulator), predictions written as pred_<OUT>_<mode>.pkl (default f4i)
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = sys.argv[2] if len(sys.argv) > 2 else "f4i"
WIN, STRIDE = 300, 150  # 5 Hz steps: 60 s windows, 30 s stride
SUB = b2_dev.SUB


def run(model, copy, stats, sub):
    with torch.no_grad():
        return model.logits(b2.to_tensor(copy["players"][sub])[None],
                            b2.to_tensor(copy["ball"][sub])[None],
                            b2.to_tensor(b2_dev.norm_ctx(copy["ctx"][sub], stats))[None],
                            b2.to_tensor(copy["dref"][sub])[None])[0].numpy()


def logits(model, copy, stats, windowed):
    """Per 10 Hz row, the model's logits (2 or 3 columns), one pass or windowed, per period."""
    model.eval()
    n = len(copy["y"])
    out = None
    period = copy["period"]
    for per in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == per)
        sub = idx[::SUB]
        if not windowed or len(sub) <= WIN:
            lg = run(model, copy, stats, sub)
        else:
            starts = list(range(0, len(sub) - WIN + 1, STRIDE))
            if starts[-1] != len(sub) - WIN:
                starts.append(len(sub) - WIN)
            lg = None
            best = np.full(len(sub), np.inf)
            for s in starts:
                part = run(model, copy, stats, sub[s:s + WIN])
                if lg is None:
                    lg = np.zeros((len(sub), part.shape[1]), np.float32)
                pos = np.arange(s, s + WIN)
                d = np.abs(pos - (s + (WIN - 1) / 2))
                # at the period edges the outermost window is the only one available
                take = d < best[pos]
                lg[pos[take]] = part[take]
                best[pos[take]] = d[take]
        if out is None:
            out = np.zeros((n, lg.shape[1]), np.float32)
        rep = np.minimum((idx - idx[0]) // SUB, len(sub) - 1)
        out[idx] = lg[rep]
    return out


sig = lambda z: 1 / (1 + np.exp(-z))


def held_out(i):
    test_dfl, test_sc = b1_dev.folds()[i]
    sets = [(f"dfl_sim_{test_dfl}", lambda w: b2_dev.dfl_copy(test_dfl, b1_dev.TEST_SEED, w)),
            (f"dfl_clean_{test_dfl}", lambda w: b2_dev.dfl_copy(test_dfl, 0, w, clean=True))]
    sets += [(f"sc_{s}", lambda w, s=s: b2_dev.sc_copy(s, w)) for s in test_sc]
    return sets


def main():
    b2.set_threads(b2_dev.THREADS)
    ref = pickle.load(open(os.path.join(b1_dev.CACHE, f"pred_b2e0{SUFFIX}_5f.pkl"), "rb"))
    per_mode = {m: {} for m in ("full", "win", "swap")}
    for tag in [m + SUFFIX for m in MEMBERS]:
        res = {m: {} for m in per_mode}
        for i in range(b1_dev.N_FOLDS):
            model, stats = load_model(os.path.join(b2_dev.MODELS, f"{tag}_f{i}_f{i}.pt"))
            for key, make in held_out(i):
                c0, c1 = make(False), make(True)
                base = {k: v for k, v in ref[key].items() if k not in ("p_alive", "p_home")}
                full = logits(model, c0, stats, False)
                win = logits(model, c0, stats, True)
                sw = logits(model, c1, stats, False)
                r = dict(base, p_alive=sig(full[:, 0]), p_home=sig(full[:, 1]))
                if full.shape[1] > 2:
                    r["p_change"] = sig(full[:, 2])
                res["full"][key] = r
                res["win"][key] = dict(base, p_alive=sig(win[:, 0]), p_home=sig(win[:, 1]))
                res["swap"][key] = dict(base, p_alive=(sig(full[:, 0]) + sig(sw[:, 0])) / 2,
                                        p_home=(sig(full[:, 1]) + 1 - sig(sw[:, 1])) / 2)
            print(tag, "fold", i, flush=True)
        for m in res:
            pickle.dump(res[m], open(os.path.join(b1_dev.CACHE, f"pred_{OUT}_{m}_{tag}.pkl"), "wb"))
            per_mode[m][tag] = res[m]
    tags = [m + SUFFIX for m in MEMBERS]
    for m, by_tag in per_mode.items():
        keys = list(by_tag[tags[0]])
        ens = {}
        for k in keys:
            r = dict(by_tag[tags[0]][k])
            r.pop("p_change", None)
            r["p_alive"] = np.mean([by_tag[t][k]["p_alive"] for t in tags], 0)
            r["p_home"] = np.mean([by_tag[t][k]["p_home"] for t in tags], 0)
            ens[k] = r
        pickle.dump(ens, open(os.path.join(b1_dev.CACHE, f"pred_{OUT}_{m}.pkl"), "wb"))
    print("done")


if __name__ == "__main__":
    main()
