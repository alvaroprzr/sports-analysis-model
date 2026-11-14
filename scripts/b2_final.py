"""Train the final B2 model on every development match (5 DFL, degraded as in development, plus
17 SkillCorner) for the mean number of steps the validated folds chose, and save its weights.

Training does not have to be reproducible (organisers' issue #11): the submission is regenerated
from the saved weights by scripts/make_submission_b2ens.py, whose inference is deterministic on CPU.

    POSS_SIM=ext_dr python scripts/b2_final.py --steps-from b2e0d_5f --config {} --out models/dr/b2_final_e0.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import b1_dev  # noqa: E402
import b2_dev  # noqa: E402
from poss import b2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-from", default=None, help="tag whose validated best steps are used")
    ap.add_argument("--steps", type=int, default=None,
                    help="number of steps (configs/b2_ens_dr.json gives each member's)")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "b2_final.pt"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    b2.set_threads(b2_dev.THREADS)
    cfg = {**b2_dev.DEFAULT, **json.loads(a.config)}
    if a.steps is not None:
        steps = a.steps
    else:
        steps_per_fold = json.load(open(os.path.join(b1_dev.CACHE, f"steps_{a.steps_from}.json")))
        steps = int(round(np.mean(list(steps_per_fold.values()))))
    win = int(cfg["win_s"] * 10 / b2_dev.SUB)
    dfl = [b2_dev.dfl_copy(m, s, w) for m in b1_dev.SPLIT["dfl"]["dev"]
           for s in range(b1_dev.N_DEGRADE) for w in (False, True)]
    sc = [b2_dev.sc_copy(m, w) for m in b1_dev.SPLIT["skillcorner"]["dev"] for w in (False, True)]
    stats = b2_dev.ctx_stats(dfl + sc)
    wd = [(ci, s) for ci, c in enumerate(dfl) for s in b2_dev.windows(c, win)]
    ws = [(ci, s) for ci, c in enumerate(sc) for s in b2_dev.windows(c, win)]
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = b2.B2(b2_dev.N_CTX, d=cfg["d"], heads=cfg["heads"], hidden=cfg["hidden"],
                  social=cfg["social"], k_near=cfg["k_near"], temporal=cfg["temporal"],
                  change_head=cfg["change_head"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / b2_dev.WARMUP))
    bs = cfg["batch"]
    n_sc = int(round(bs * b2_dev.SC_SHARE))
    recent = []
    for step in range(steps):
        pd_ = [wd[k] for k in rng.choice(len(wd), bs - n_sc, replace=False)]
        ps_ = [ws[k] for k in rng.choice(len(ws), n_sc, replace=False)]
        loss = b2_dev.loss_dfl(model, *b2_dev.batch(dfl, pd_, stats, win), aux=cfg["aux"],
                               cfg=cfg) + \
            b1_dev.SC_TOTAL_WEIGHT * b2_dev.loss_sc(model, *b2_dev.batch(sc, ps_, stats, win))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        recent.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{steps} loss {np.mean(recent[-50:]):.4f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": cfg, "stats": stats, "steps": steps}, a.out)
    print("saved", a.out, "steps", steps)


if __name__ == "__main__":
    main()
