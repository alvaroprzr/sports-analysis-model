"""SkillCorner open data (A-League 2024/25, broadcast tracking at 10 fps, MIT) in the common format.

Labels come from SkillCorner's own phases of play (`<id>_phases_of_play.csv`): every frame inside
a phase belongs to `team_in_possession_id`; frames between phases (interruptions) are taken as
dead. That dead-ball label is an approximation -- the dataset has no per-frame ball status -- so
scores on it are reported apart from the team scores.

Player positions include SkillCorner's extrapolation of players outside the camera view. The
flag is kept in `meta["detected"]` so experiments can drop extrapolated positions.
"""

import json
import os

import numpy as np
import pandas as pd

from .match import Match


def load(match_dir):
    mid = os.path.basename(os.path.normpath(match_dir))
    info = json.load(open(os.path.join(match_dir, f"{mid}_match.json")))
    home_id = info["home_team"]["id"]
    slot = {}
    counts = [0, 0]
    for p in info["players"]:
        side = 0 if p["team_id"] == home_id else 1
        slot[p["id"]] = (side, counts[side])
        counts[side] += 1

    lines = open(os.path.join(match_dir, f"{mid}_tracking_extrapolated.jsonl")).read().splitlines()
    rows = [json.loads(s) for s in lines]
    n = max(r["frame"] for r in rows) + 1
    home = np.full((n, counts[0], 2), np.nan, np.float32)
    away = np.full((n, counts[1], 2), np.nan, np.float32)
    det = [np.zeros((n, counts[0]), bool), np.zeros((n, counts[1]), bool)]
    ball = np.full((n, 3), np.nan, np.float32)
    period = np.zeros(n, np.int8)
    for r in rows:
        k = r["frame"]
        period[k] = r["period"] or 0
        b = r["ball_data"]
        if b and b["x"] is not None:
            ball[k] = (b["x"], b["y"], b["z"] if b["z"] is not None else np.nan)
        for p in r["player_data"]:
            side, i = slot[p["player_id"]]
            (home if side == 0 else away)[k, i] = (p["x"], p["y"])
            det[side][k, i] = bool(p["is_detected"])

    alive = np.where(period > 0, 0, -1).astype(np.int8)
    team = np.full(n, -1, np.int8)
    phases = pd.read_csv(os.path.join(match_dir, f"{mid}_phases_of_play.csv"))
    for start, end, tid in phases[["frame_start", "frame_end", "team_in_possession_id"]].values:
        alive[start:end + 1] = 1
        team[start:end + 1] = 0 if tid == home_id else 1
    return Match(
        name=mid, source="skillcorner", fps=10.0,
        pitch=(float(info["pitch_length"]), float(info["pitch_width"])),
        home=home, away=away, ball=ball, period=period, alive=alive, team=team,
        meta={"detected_home": det[0], "detected_away": det[1]},
    )
