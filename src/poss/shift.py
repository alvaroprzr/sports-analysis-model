"""Boundary shifts after decoding (no training): every boundary of a given type in the decoded
working-rate sequence is moved by a fixed number of rows, before the decisions are held over the
native frames. Positive = later (the earlier segment is extended).

Types (the label before -> the label after):
    team     H -> A or A -> H
    d_start  H or A -> D
    d_end    D -> H or A

Boundaries are moved only inside a period; every segment keeps at least one row, so no segment
disappears and no new one is created.
"""

import numpy as np

H, A, D = 0, 1, 2
TYPES = ("team", "d_start", "d_end")
RATE = 10


def boundary_type(prev, nxt):
    if prev in (H, A) and nxt in (H, A):
        return "team"
    if prev in (H, A) and nxt == D:
        return "d_start"
    if prev == D and nxt in (H, A):
        return "d_end"
    return None


def boundaries(codes):
    """(index of the first row of the new segment, type) for every change inside the sequence."""
    b = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    return [(int(i), boundary_type(int(codes[i - 1]), int(codes[i]))) for i in b]


def _shift_segment(codes, rows):
    n = len(codes)
    bs = boundaries(codes)
    if not bs:
        return codes
    labels = [int(codes[0])] + [int(codes[i]) for i, _ in bs]
    want = [i + (rows.get(t, 0) if t else 0) for i, t in bs]
    new = list(want)
    for j in range(len(new)):  # left to right: at least one row after the previous boundary
        lo = (new[j - 1] if j else 0) + 1
        new[j] = max(new[j], lo)
    for j in range(len(new) - 1, -1, -1):  # right to left: at least one row before the next one
        hi = (new[j + 1] if j + 1 < len(new) else n) - 1
        new[j] = min(new[j], hi)
    out = np.empty(n, codes.dtype)
    edges = [0] + new + [n]
    for j, lab in enumerate(labels):
        out[edges[j]:edges[j + 1]] = lab
    return out


def apply(codes, period, shifts_s, rate=RATE):
    """codes, period at the working rate; shifts_s {type: seconds}."""
    rows = {t: int(round(shifts_s.get(t, 0.0) * rate)) for t in TYPES}
    if not any(rows.values()):
        return codes
    out = codes.copy()
    for per in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == per)
        out[idx] = _shift_segment(codes[idx], rows)
    return out


def signed_offsets(gt, pred, fps, max_s=2.0):
    """Per type, the signed offset (s, predicted minus true) of each true boundary to the nearest
    predicted boundary of the same type within max_s; frames marked X on either side are cut."""
    out = {t: [] for t in TYPES}
    pb = {t: np.array([i for i, ty in boundaries(pred) if ty == t]) for t in TYPES}
    for i, t in boundaries(gt):
        if t is None or len(pb[t]) == 0:
            continue
        d = pb[t] - i
        j = int(np.argmin(np.abs(d)))
        if abs(d[j]) <= max_s * fps:
            out[t].append(d[j] / fps)
    return out
