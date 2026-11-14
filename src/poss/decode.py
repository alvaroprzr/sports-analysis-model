"""Segmentation of per-frame H / A / D probabilities with a three-state linear chain.

Transition scores are three costs (nats): changing team while alive, entering a stoppage and
leaving it. They set the granularity of the output (how many possessions), which is the decision
the sources disagree on most, so they are chosen on the development matches of both sources
(see scripts/b1_dev.py), never from one source's optimum.
"""

import numpy as np

H, A, D = 0, 1, 2


def transition_matrix(c_team, c_in, c_out):
    T = np.zeros((3, 3))
    T[H, A] = T[A, H] = -c_team
    T[H, D] = T[A, D] = -c_in
    T[D, H] = T[D, A] = -c_out
    return T


def viterbi(logp, trans):
    """Most likely state sequence for emissions logp (T, 3) under transition scores (3, 3).

    Plain scalar loop: with three states it is several times faster than numpy per step."""
    n = len(logp)
    lp = logp.tolist()
    tr = trans.tolist()
    back = bytearray(3 * n)
    s0, s1, s2 = lp[0]
    for t in range(1, n):
        e0, e1, e2 = lp[t]
        new = [0.0, 0.0, 0.0]
        for j in range(3):
            a, b, c = s0 + tr[0][j], s1 + tr[1][j], s2 + tr[2][j]
            if a >= b and a >= c:
                new[j] = a
            elif b >= c:
                new[j], back[3 * t + j] = b, 1
            else:
                new[j], back[3 * t + j] = c, 2
        s0, s1, s2 = new[0] + e0, new[1] + e1, new[2] + e2
    path = np.zeros(n, np.int8)
    last = [s0, s1, s2]
    k = last.index(max(last))
    path[-1] = k
    for t in range(n - 1, 0, -1):
        k = back[3 * t + k]
        path[t - 1] = k
    return path


def decode(p_alive, p_home, period, costs, eps=1e-4):
    """Codes at the working rate. Emissions are log P(state | x) from the two classifiers;
    frames outside play are left as X (3)."""
    pa = np.clip(p_alive, eps, 1 - eps)
    ph = np.clip(p_home, eps, 1 - eps)
    logp = np.log(np.column_stack([pa * ph, pa * (1 - ph), 1 - pa]))
    trans = transition_matrix(*costs)
    out = np.full(len(period), 3, np.int8)
    for per in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == per)
        out[idx] = viterbi(logp[idx], trans)
    return out


def to_native(codes10, near, n_native):
    """Hold each working-rate decision over the native frames closest to it."""
    out = np.full(n_native, 3, np.int8)
    f = np.arange(n_native)
    j = np.clip(np.searchsorted(near, f, side="left"), 0, len(near) - 1)
    jm = np.clip(j - 1, 0, len(near) - 1)
    pick = np.where(np.abs(near[jm] - f) <= np.abs(near[j] - f), jm, j)
    out[:] = codes10[pick]
    return out
