"""Match structure published by the organisers, applied to the decoded labels of BAR-ATH.

FAQ.md and DATA.md: the second half is graded from video frame 110,808 but does not kick off
until 113,742, and "those 2,934 frames are graded and all of them are D". That is a published
boundary of the match (like the half limits), not a label learnt or read from the ground truth.
Graded frames must also carry an answer: an X inside a graded half (it can appear at the last
frame of a half, where the tracking ends one frame early) takes the previous label.
"""

import numpy as np

from games import half_video_blocks

from . import dataset

# FAQ.md, "One trap worth knowing"
SECOND_HALF_GRADED_FROM = 110808
SECOND_HALF_KICKOFF = 113742


def apply(codes, game):
    """codes: int labels per video frame (dataset.H / A / D / X). Returns a corrected copy and
    the number of frames changed by each rule."""
    out = np.asarray(codes).copy()
    pre = out[SECOND_HALF_GRADED_FROM:SECOND_HALF_KICKOFF]
    n_pre = int((pre != dataset.D).sum())
    out[SECOND_HALF_GRADED_FROM:SECOND_HALF_KICKOFF] = dataset.D
    n_x = 0
    for _, first, last in half_video_blocks(game):
        for i in range(first, last + 1):
            if out[i] == dataset.X:
                out[i] = out[i - 1] if i > first and out[i - 1] != dataset.X else dataset.D
                n_x += 1
    return out, {"pre_kickoff_to_D": n_pre, "graded_X_filled": n_x}
