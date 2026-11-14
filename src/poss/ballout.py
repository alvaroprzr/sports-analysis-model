"""Ball-out evidence for the decoder (F5): rows where the visible ball has been outside the pitch
lines for a while are given P(alive) ~ 0 before decoding, so the chain itself re-places the
neighbouring possessions and the team that restarts.

A 10 Hz row counts as "out" when the ball is visible, more than `min_out_m` beyond a touchline or
goal line, and slower than MAX_SPEED (a fast ball is in flight, where the projection of a high
ball can fall outside the lines). Only runs of consecutive out rows lasting at least `min_s`
seconds are used. Everything comes from the ball token of src/poss/tokens.py (visible flag,
velocity, distance beyond a line), the same on every source.
"""

import numpy as np

# Params
MAX_SPEED = 25.0  # m/s, fixed before validation (05/11)
P_DEAD = 1e-4
RATE = 10
VIS, VX, VY, BEYOND = 4, 2, 3, 7  # columns of the ball token
SCALE_V, SCALE_BEYOND = 10.0, 5.0


def out_rows(ball, min_out_m, min_s):
    b = ball.astype(np.float64)
    speed = np.hypot(b[:, VX], b[:, VY]) * SCALE_V
    out = (b[:, VIS] > 0.5) & (b[:, BEYOND] * SCALE_BEYOND > min_out_m) & (speed <= MAX_SPEED)
    keep = np.zeros(len(out), bool)
    t = np.flatnonzero(np.diff(np.r_[0, out.astype(np.int8), 0]))
    for a, e in zip(t[::2], t[1::2]):
        if (e - a) / RATE >= min_s:
            keep[a:e] = True
    return keep


def apply(p_alive, ball, min_out_m, min_s):
    p = p_alive.copy()
    p[out_rows(ball, min_out_m, min_s)] = P_DEAD
    return p
