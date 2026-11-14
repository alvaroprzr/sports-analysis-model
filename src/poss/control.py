"""Control reading of the DFL possession labels, from the IDSSE event files (labels only).

CHALLENGE.md defines a change of possession as the other team gaining *controlled* possession: a
touch without control, a deflection or an aerial duel won without subsequent control do not count,
and a loose ball stays with the last team that had it. The DFL BallPossession flag changes on
every touch. This module rewrites the flag so that a run of the other team counts as a new
possession only when that team shows control. The rule was fixed on 5 November, before any score
was computed:

1. Runs = maximal runs of one team in the flag while the ball is alive. A run that starts right
   after a stoppage keeps its team (play restarts with a clear possessor).
2. Event clock -> tracking clock per half: t = 0 at the KickOff event and at the first live frame
   of the half; then a fine shift s in [-2, +2] s (step 0.04 s) that minimises the median distance
   between the ball and the positions of passes, crosses and shots.
3. In order, each run of team T is compared with the team of the possession as already
   corrected. If T differs, the run is a new possession only if T has a control event in
   [start - 1.5 s, end + 1.5 s]: a pass or cross, a shot, a ball claim, a set piece (throw-in,
   free kick, corner, goal kick, kick-off, penalty), or a ground duel won by T keeping the ball
   (WinnerResult ballClaimed, ballControlRetained or dribbledAround).
4. Without such an event, the indicators CHALLENGE.md suggests decide: the run lasts at least 3 s,
   or T makes at least 3 passes in a row inside it (passes are control events, so in practice
   only the duration can decide here; both are counted).
5. Otherwise the run takes the corrected possession's team.

Delete events carry no team or content and are ignored; so are touches (OtherBallAction), fouls
and aerial duels. Events are never a model input.
"""

import glob
import os
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

# Params
EVENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "external", "dfl_events_raw")
WINDOW_S = 1.5
MIN_RUN_S = 3.0
MIN_PASSES = 3
SHIFT_RANGE_S = 2.0
SHIFT_STEP_S = 0.04
SET_PIECES = ("ThrowIn", "FreeKick", "CornerKick", "GoalKick", "KickOff", "Penalty")
DUEL_WON = ("ballClaimed", "ballControlRetained", "dribbledAround")


def _time(s):
    return datetime.fromisoformat(s).timestamp()


def _files(mid):
    ev = glob.glob(os.path.join(EVENT_DIR, f"DFL_03_02_events_raw_*_DFL-MAT-{mid}.xml"))
    info = glob.glob(os.path.join(EVENT_DIR, f"DFL_02_01_matchinformation_*_DFL-MAT-{mid}.xml"))
    assert len(ev) == 1 and len(info) == 1, mid
    return ev[0], info[0]


def load_events(mid):
    """Control events of a match: list of dicts with t (s, wall clock), team (0 home / 1 away),
    kind ('pass', 'shot', 'claim', 'set', 'duel'), x, y (event coordinates, metres, corner origin)
    and whether it locates the ball (passes, crosses, shots). Also the kick-off time of each half."""
    ev_path, info_path = _files(mid)
    general = ET.parse(info_path).getroot().find(".//General")
    home = general.get("HomeTeamId")
    out, kickoff = [], {}
    for e in ET.parse(ev_path).getroot().findall("Event"):
        if e.find("Delete") is not None:
            continue
        t = _time(e.get("EventTime"))
        x, y = float(e.get("X-Position", "nan")), float(e.get("Y-Position", "nan"))
        for c in e:
            kind, team = None, None
            if c.tag == "KickOff":
                # half starts only: kick-offs after a goal carry no GameSection
                half = {"firstHalf": 1, "secondHalf": 2}.get(c.get("GameSection"))
                if half is not None:
                    kickoff[half] = min(t, kickoff.get(half, t))
            if c.tag in SET_PIECES:
                play = c.find("Play")
                team = c.get("Team") or (play.get("Team") if play is not None else None)
                kind = "set"
            elif c.tag == "Play":
                team, kind = c.get("Team"), "pass"
            elif c.tag == "ShotAtGoal":
                team, kind = c.get("Team"), "shot"
            elif c.tag == "BallClaiming":
                team, kind = c.get("Team"), "claim"
            elif c.tag == "TacklingGame":
                if c.get("Type") == "ground" and c.get("WinnerResult") in DUEL_WON:
                    team, kind = c.get("WinnerTeam"), "duel"
            if kind is None or team is None:
                continue
            out.append(dict(t=t, team=0 if team == home else 1, kind=kind, x=x, y=y,
                            locates=kind in ("pass", "shot")))
    return out, kickoff


def half_clock(m):
    """Seconds since the first live frame of its half, per frame (NaN outside halves 1, 2)."""
    t = np.full(m.n_frames, np.nan)
    for h in (1, 2):
        idx = np.flatnonzero((m.period == h) & (m.alive == 1))
        if len(idx):
            sel = np.flatnonzero(m.period == h)
            t[sel] = (sel - idx[0]) / m.fps
    return t


def fit_shift(m, events, kickoff):
    """Per half: (shift in s, y sign, median distance in m) aligning event to tracking clock."""
    clock = half_clock(m)
    L, W = m.pitch
    res = {}
    for h in (1, 2):
        sel = np.flatnonzero(m.period == h)
        th = clock[sel]
        loc = [e for e in events if e["locates"] and _half_of(e, kickoff) == h]
        if not loc or not len(sel):
            continue
        te = np.array([e["t"] - kickoff[h] for e in loc])
        ex = np.array([e["x"] for e in loc]) - L / 2
        ey0 = np.array([e["y"] for e in loc]) - W / 2
        best = None
        for sign in (1, -1):
            ey = sign * ey0
            for s in np.arange(-SHIFT_RANGE_S, SHIFT_RANGE_S + 1e-9, SHIFT_STEP_S):
                k = np.searchsorted(th, te + s)
                ok = (k > 0) & (k < len(sel))
                fr = sel[np.clip(k, 0, len(sel) - 1)][ok]
                b = m.ball[fr, :2]
                d = np.hypot(b[:, 0] - ex[ok], b[:, 1] - ey[ok])
                med = np.nanmedian(d)
                if best is None or med < best[2]:
                    best = (round(float(s), 2), sign, float(med))
        res[h] = best
    return res


def _half_of(e, kickoff):
    return 2 if 2 in kickoff and e["t"] >= kickoff[2] else 1


def relabel(m, events, kickoff, shifts, log=None):
    """Control reading of m.team (alive frames only). Returns the new team array and counts of
    how each run that would change possession was decided. If `log` is a list, one record per
    would-be change is appended: (duration s, decision, the run ends in a stoppage)."""
    clock = half_clock(m)
    fps = m.fps
    # control events on the tracking clock, per half and team: (frame time, kind)
    ev = {(h, tm): [] for h in (1, 2) for tm in (0, 1)}
    for e in events:
        h = _half_of(e, kickoff)
        if h not in shifts:
            continue
        ev[(h, e["team"])].append((e["t"] - kickoff[h] + shifts[h][0], e["kind"]))
    for k in ev:
        ev[k].sort()
    team = m.team.copy()
    alive = m.alive == 1
    counts = dict(runs=0, changes=0, by_event=0, by_duration=0, by_passes=0, reverted=0,
                  restart=0)
    for h in (1, 2):
        idx = np.flatnonzero((m.period == h) & alive & (m.team >= 0))
        if not len(idx):
            continue
        # runs of the flag inside live play; a gap in idx is a stoppage
        brk = np.flatnonzero((np.diff(idx) > 1) | (np.diff(m.team[idx]) != 0)) + 1
        starts = np.r_[0, brk]
        ends = np.r_[brk, len(idx)]
        current = None
        for a, b in zip(starts, ends):
            fr = idx[a:b]
            t_run = m.team[fr[0]]
            counts["runs"] += 1
            after_stop = a == 0 or idx[a] - idx[a - 1] > 1
            if after_stop or current is None:
                current = t_run
                counts["restart"] += 1
                continue
            if t_run == current:
                continue
            counts["changes"] += 1
            t0, t1 = clock[fr[0]], clock[fr[-1]]
            ends_in_stop = b == len(idx) or idx[b] - idx[b - 1] > 1
            before = dict(counts)
            lst = ev[(h, int(t_run))]
            inside = [k for (t, k) in lst if t0 - WINDOW_S <= t <= t1 + WINDOW_S]
            if inside:
                current = t_run
                counts["by_event"] += 1
            elif (t1 - t0 + 1 / fps) >= MIN_RUN_S:
                current = t_run
                counts["by_duration"] += 1
            elif _passes_in_a_row(lst, ev[(h, 1 - int(t_run))], t0, t1) >= MIN_PASSES:
                current = t_run
                counts["by_passes"] += 1
            else:
                team[fr] = current
                counts["reverted"] += 1
            if log is not None:
                decision = next(k for k in ("by_event", "by_duration", "by_passes", "reverted")
                                if counts[k] != before[k])
                log.append((t1 - t0 + 1 / fps, decision, bool(ends_in_stop)))
    return team, counts


def _passes_in_a_row(own, other, t0, t1):
    """Longest streak of passes of the team inside [t0, t1] not interrupted by an event of the
    other team."""
    seq = sorted([(t, 0, k) for t, k in own if t0 <= t <= t1] +
                 [(t, 1, k) for t, k in other if t0 <= t <= t1])
    best = cur = 0
    for _, who, k in seq:
        if who == 0 and k == "pass":
            cur += 1
            best = max(best, cur)
        elif who == 1:
            cur = 0
    return best
