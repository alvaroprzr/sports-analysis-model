"""Scores on the external matches with the organisers' metric (evaluation/possession_metrics.py).

- DFL: the official spellF1 and macroF1 over the whole match at 25 fps.
- SkillCorner: its dead-ball label is only approximate, so its headline score is the spellF1 of
  possessions inside live play: frames the reference marks dead are ungraded (X) and a predicted
  D is treated as an abstention (forward-filled, as the official scorer does with X). The full
  score with its approximate D is reported separately.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
SC_WEIGHT = 0.5  # selection score = (DFL + 0.5 * SkillCorner live) / 1.5, agreed with the team


def full_scores(gt, pred):
    runs = pm.valid_runs(gt)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    sp = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    return {"spellF1": sp["mean_f1"], "macroF1": pm.macro_f1(cm3), "n_pred": sp["n_pred"],
            "n_gt": sp["n_gt"]}


def live_scores(gt, pred):
    g = gt.copy()
    g[g == pm.D] = pm.X
    p = pred.copy()
    p[p == pm.D] = pm.X
    runs = pm.valid_runs(g)
    sp = pm.spell_scores(g, pm.fill_abstentions(p, runs), runs)
    return {"spellF1_live": sp["mean_f1"], "n_pred_live": sp["n_pred"], "n_gt_live": sp["n_gt"]}


def combined(dfl, sc):
    """Selection score from per-source means; both are always reported separately too."""
    return (dfl + SC_WEIGHT * sc) / (1 + SC_WEIGHT)


# Params
DURATION_BINS_S = (0, 2, 4, 8, 16, np.inf)


def duration_counts(gt, pred, fps=pm.FPS):
    """Per duration bin and IoU threshold: matched / total ground-truth spells (by true duration)
    and matched / total predicted spells (by predicted duration), with the official greedy
    one-to-one matching. Sum the counts over matches, then `duration_table`."""
    runs = pm.valid_runs(gt)
    g = pm.possession_spells(gt, runs)
    p = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    ov = pm.spell_overlaps(g, p)
    bins = np.array(DURATION_BINS_S)
    gb = np.searchsorted(bins, [(b - a) / fps for a, b, _ in g], side="right") - 1
    pb = np.searchsorted(bins, [(b - a) / fps for a, b, _ in p], side="right") - 1
    nb, nt = len(bins) - 1, len(pm.SPELL_IOUS)
    out = {k: np.zeros((nt, nb)) for k in ("gt_hit", "gt_all", "pred_hit", "pred_all")}
    for ti, thr in enumerate(pm.SPELL_IOUS):
        gf = np.zeros(len(g), bool)
        pf = np.zeros(len(p), bool)
        for iou, i, j in ov:
            if iou < thr:
                break
            if gf[i] or pf[j]:
                continue
            gf[i] = pf[j] = True
        out["gt_hit"][ti] = np.bincount(gb[gf], minlength=nb)[:nb] if len(g) else 0
        out["gt_all"][ti] = np.bincount(gb, minlength=nb)[:nb] if len(g) else 0
        out["pred_hit"][ti] = np.bincount(pb[pf], minlength=nb)[:nb] if len(p) else 0
        out["pred_all"][ti] = np.bincount(pb, minlength=nb)[:nb] if len(p) else 0
    return out


def duration_table(counts):
    """Recall (by true duration), precision (by predicted duration) and their F1, averaged over
    the IoU thresholds, from summed `duration_counts`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = counts["gt_hit"] / counts["gt_all"]
        prec = counts["pred_hit"] / counts["pred_all"]
        f1 = 2 * rec * prec / (rec + prec)
    return dict(recall=np.nanmean(rec, 0), precision=np.nanmean(prec, 0),
                f1=np.nanmean(np.nan_to_num(f1), 0), n_gt=counts["gt_all"][0],
                n_pred=counts["pred_all"][0])
