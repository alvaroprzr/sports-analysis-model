"""Cross-machine reproducibility check: the same steps as scripts/make_submission_b2ens.py, printing
a SHA-256 per stage so that two machines can find where their outputs part:

    a. inputs of BAR-ATH: physics rows (F), player tokens, ball tokens, reference distances, rows
    b. each member's two probabilities before any rounding
    c. their mean
    d. the mean rounded to 1e-4 (what the decoder receives)
    e. decoded codes (10 Hz, then native frames with the published match structure)
    f. the CSV bytes

Floating-point stages are also hashed after rounding to fewer decimals, which tells how large a
difference is (if two machines agree at 1e-6 but not exactly, it is last-bit noise). With --float64
the members run in double precision. With --dump DIR the stage arrays are saved (local use);
with --compare DIR the maximum absolute difference of every stage against the arrays another
machine dumped there is printed (only those numbers, no data).

Variants to find which one makes two machines agree:
    --no-mkldnn    torch.backends.mkldnn.enabled = False
    --legacy-topk  E1's nearest-player selection with torch.topk (the default is a stable sort,
                   ties broken by slot index, identical on every CPU)
    --windowed     overlapping 60 s windows (30 s stride, each step from its most central window)
For each member the encoder output (before the GRU) and the GRU output are hashed too.

    python scripts/repro_stages.py [--float64] [--no-mkldnn] [--stable-topk] [--windowed]
                                   [--threads N] [--dump DIR] [--compare DIR]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import b2_dev  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from make_submission_b2ens import LABELS, ROUND, load_model  # noqa: E402
from poss import b2, ballout, barath, dataset, decode, decode2, published, tokens  # noqa: E402

DECIMALS = (6, 4, 3)


def h(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()).hexdigest()[:16]


COMPARE = None


def report(name, x, dump=None):
    x = np.asarray(x)
    if COMPARE:
        path = os.path.join(COMPARE, name.replace(" ", "_") + ".npy")
        if os.path.exists(path):
            y = np.load(path)
            if y.shape == x.shape:
                dd = np.abs(x.astype(np.float64) - y.astype(np.float64))
                dd = dd[np.isfinite(dd)]
                print(f"   vs {COMPARE}: max |diff| {dd.max() if len(dd) else 0:.3e}, "
                      f"values differing {int((dd > 0).sum())} of {x.size}")
    cells = [f"exact {h(x)}"]
    if np.issubdtype(x.dtype, np.floating):
        for d in DECIMALS:
            cells.append(f"1e-{d} {h(np.round(x.astype(np.float64), d))}")
        cells.append(f"nan {int(np.isnan(x).sum())}")
    print(f"{name:28s} {x.dtype} {tuple(x.shape)}  " + "  ".join(cells), flush=True)
    if dump:
        np.save(os.path.join(dump, name.replace(" ", "_") + ".npy"), x)


WIN, STRIDE = 300, 150  # 5 Hz steps


def forward(model, copy, stats, sub, f64):
    """Encoder output, GRU output and logits for the 5 Hz steps `sub` of one period."""
    dt = np.float64 if f64 else np.float32
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=dt))
    P, Bl = t(copy["players"][sub]), t(copy["ball"][sub])
    C, R = t(b2_dev.norm_ctx(copy["ctx"][sub], stats)), t(copy["dref"][sub])
    with torch.no_grad():
        z = torch.cat([model.enc(P[i:i + 4096], Bl[i:i + 4096], C[i:i + 4096], R[i:i + 4096])
                       for i in range(0, len(sub), 4096)])
        hseq = model.gru(z[None])[0]
        lg = model.head(hseq)[0]
    return z.numpy(), hseq[0].numpy(), lg.numpy()


def predict(model, copy, stats, f64, windowed):
    """p_alive, p_home at the 10 Hz rows, and the encoder / GRU outputs of the one-pass run."""
    model = (model.double() if f64 else model).eval()
    n = len(copy["y"])
    pa, ph = np.full(n, 0.5), np.full(n, 0.5)
    zs, hs = [], []
    period = copy["period"]
    for per in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == per)
        sub = idx[::b2_dev.SUB]
        z, hseq, lg = forward(model, copy, stats, sub, f64)
        zs.append(z)
        hs.append(hseq)
        if windowed and len(sub) > WIN:
            starts = list(range(0, len(sub) - WIN + 1, STRIDE))
            if starts[-1] != len(sub) - WIN:
                starts.append(len(sub) - WIN)
            best = np.full(len(sub), np.inf)
            for s0 in starts:
                part = forward(model, copy, stats, sub[s0:s0 + WIN], f64)[2]
                pos = np.arange(s0, s0 + WIN)
                d = np.abs(pos - (s0 + (WIN - 1) / 2))
                take = d < best[pos]
                lg[pos[take]] = part[take]
                best[pos[take]] = d[take]
        rep = np.minimum((idx - idx[0]) // b2_dev.SUB, len(sub) - 1)
        pa[idx] = 1 / (1 + np.exp(-lg[rep, 0].astype(np.float64)))
        ph[idx] = 1 / (1 + np.exp(-lg[rep, 1].astype(np.float64)))
    return pa, ph, np.concatenate(zs), np.concatenate(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "b2_ens_dr.json"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--no-mkldnn", action="store_true")
    ap.add_argument("--legacy-topk", action="store_true",
                    help="E1's selection with torch.topk (ties broken differently per CPU)")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    global COMPARE
    COMPARE = a.compare
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
    b2.set_threads(a.threads)
    if a.no_mkldnn:
        torch.backends.mkldnn.enabled = False
    print("versions:", json.dumps({"python": sys.version.split()[0], "numpy": np.__version__,
                                   "torch": torch.__version__, "pandas": pd.__version__}))
    print("cpu capability:", torch.backends.cpu.get_cpu_capability(),
          "| mkldnn available:", torch.backends.mkldnn.is_available(),
          "enabled:", torch.backends.mkldnn.enabled, "| float64:", a.float64,
          "| legacy topk:", a.legacy_topk, "| windowed:", a.windowed)
    conf = json.load(open(a.config))
    bar = barath.load()
    report("a0 ball (raw)", bar.ball)
    report("a0 home (raw)", bar.home)
    pl, bl, near, dref = tokens.tokens(bar)
    F, _, _, period, near_rows = dataset.rows(bar)
    report("a1 rows F", F, a.dump)
    report("a2 players tokens", pl, a.dump)
    report("a3 ball tokens", bl, a.dump)
    report("a4 dref", dref, a.dump)
    report("a5 near", near)
    report("a6 period", period)
    copy = dict(players=pl, ball=bl, dref=dref, ctx=F[:, :b2_dev.N_CTX], period=period,
                y=np.zeros(len(period), np.int8))
    swap = conf.get("team_swap", False)
    if swap:
        sw = dataset.swap_teams(bar)
        pl2, bl2, _, dref2 = tokens.tokens(sw)
        F2 = dataset.rows(sw)[0]
        report("a7 swapped players tokens", pl2)
        report("a8 swapped rows F", F2)
        copy2 = dict(players=pl2, ball=bl2, dref=dref2, ctx=F2[:, :b2_dev.N_CTX], period=period,
                     y=np.zeros(len(period), np.int8))
    pas, phs = [], []
    for name in sorted(conf["members"]):
        model, stats = load_model(os.path.join(ROOT, conf["members"][name]["weights"]))
        model.enc.stable_topk = not a.legacy_topk
        pa, ph, z, hseq = predict(model, copy, stats, a.float64, a.windowed)
        report(f"b {name} encoder", z, a.dump)
        report(f"b {name} gru", hseq, a.dump)
        if swap:
            pa2, ph2, _, _ = predict(model, copy2, stats, a.float64, a.windowed)
            report(f"b {name} swapped p_alive", pa2, a.dump)
            report(f"b {name} swapped p_home", ph2, a.dump)
            pa, ph = (pa + pa2) / 2, (ph + 1 - ph2) / 2
        report(f"b {name} p_alive", pa, a.dump)
        report(f"b {name} p_home", ph, a.dump)
        pas.append(pa)
        phs.append(ph)
    pa, ph = np.mean(pas, 0), np.mean(phs, 0)
    bo = conf.get("ball_out")
    if bo:
        pa = ballout.apply(pa, bl, bo["min_out_m"], bo["min_s"])
    report("c mean p_alive", pa)
    report("c mean p_home", ph)
    pa, ph = np.round(pa, ROUND), np.round(ph, ROUND)
    report("d rounded p_alive", pa, a.dump)
    report("d rounded p_home", ph, a.dump)
    k = int(conf.get("min_possession_rows", 1))
    codes10 = (decode.decode(pa, ph, period, tuple(conf["costs"])) if k == 1
               else decode2.decode(pa, ph, period, tuple(conf["costs"]), k=k))
    report("e codes 10 Hz", codes10, a.dump)
    codes = decode.to_native(codes10, near, bar.n_frames)
    codes, _ = published.apply(codes, GAME)
    report("e codes native", codes, a.dump)
    n = GAMES[GAME]["n_video_frames"]
    csv = pd.DataFrame({"frame": np.arange(n), "label": LABELS[codes]}).to_csv(
        index=False, lineterminator="\n").encode()
    print("f csv sha256", hashlib.sha256(csv).hexdigest())
    print("counts", {k: int((LABELS[codes] == k).sum()) for k in "HADX"})


if __name__ == "__main__":
    main()
