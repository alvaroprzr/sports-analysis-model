"""Decoder variants for F4 (same emissions as decode.decode, numba Viterbi over an expanded chain):

- variable team-switch cost: the H <-> A transition at row t costs c_team(t) instead of c_team;
- minimum possession duration (an explicit-duration approximation): H and A are each expanded
  into k sub-states that must be passed in order, so a possession lasts at least k rows
  (k / 10 s); k = 1 is the plain decoder. Stoppages keep one state.

decode.py is left untouched (it is used by the submitted pipeline).
"""

import numba
import numpy as np

H, A, D = 0, 1, 2
NEG = -1e18


@numba.njit(cache=True)
def _viterbi(em, trans, switch, extra):
    """em (T, S) emissions per expanded state; trans (S, S) scores; switch (S, S) 1 where the
    transition changes team; extra (T,) added to switching transitions entering row t."""
    n, S = em.shape
    back = np.zeros((n, S), np.int16)
    s = em[0].copy()
    new = np.empty(S)
    for t in range(1, n):
        for j in range(S):
            best, arg = NEG, 0
            for i in range(S):
                v = s[i] + trans[i, j] + switch[i, j] * extra[t]
                if v > best:
                    best, arg = v, i
            new[j] = best + em[t, j]
            back[t, j] = arg
        s[:] = new
    path = np.zeros(n, np.int16)
    k = 0
    for j in range(S):
        if s[j] > s[k]:
            k = j
    path[n - 1] = k
    for t in range(n - 1, 0, -1):
        k = back[t, k]
        path[t - 1] = k
    return path


def _chain(costs, k):
    """Expanded states: H_1..H_k, A_1..A_k, D. Returns (trans, switch, label per state)."""
    c_team, c_in, c_out = costs
    S = 2 * k + 1
    T = np.full((S, S), NEG)
    W = np.zeros((S, S))
    lab = np.array([H] * k + [A] * k + [D])
    dd = 2 * k
    for team in (0, 1):
        o = team * k
        other = (1 - team) * k
        for i in range(k):
            T[o + i, o + i] = 0.0 if i == k - 1 else NEG  # only the last sub-state may stay
            if i < k - 1:
                T[o + i, o + i + 1] = 0.0
        T[o + k - 1, other] = -c_team
        W[o + k - 1, other] = 1.0
        T[o + k - 1, dd] = -c_in
        T[dd, o] = -c_out
    T[dd, dd] = 0.0
    if k == 1:
        T[0, 0] = T[1, 1] = 0.0
    return T, W, lab


def decode(p_alive, p_home, period, costs, k=1, team_cost=None, eps=1e-4):
    """Codes at the working rate. team_cost: per-row H <-> A cost replacing costs[0]."""
    pa = np.clip(p_alive, eps, 1 - eps)
    ph = np.clip(p_home, eps, 1 - eps)
    logp = np.log(np.column_stack([pa * ph, pa * (1 - ph), 1 - pa]))
    trans, switch, lab = _chain(costs, k)
    out = np.full(len(period), 3, np.int8)
    for per in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == per)
        em = np.ascontiguousarray(logp[idx][:, lab])
        extra = (np.zeros(len(idx)) if team_cost is None
                 else costs[0] - np.asarray(team_cost, np.float64)[idx])
        out[idx] = lab[_viterbi(em, trans, switch, np.ascontiguousarray(extra))]
    return out
