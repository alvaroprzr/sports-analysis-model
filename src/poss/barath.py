"""The provided match (Metrica ATD, FIFA/EPTS export) in the common format. No labels.

Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>" -- player slots, then the ball, each as
fractions of the pitch (origin top-left, y down) plus a speed channel. frame_id counts from 1,
video frames (and submission rows) from 0. The speed channel is not used: positions are enough
and the ball speed is mostly NaN in this export.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .match import Match

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402


def team_of_slot(meta_path):
    """0 for every slot of the first listed team (home, per starter/baseline.py), 1 otherwise."""
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")], np.int8)


def load(game=GAME, data_dir=None, source="tactical"):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    length, width = GAMES[game]["pitch"]
    side = team_of_slot(meta_path)
    n = sum(1 for _ in open(raw_path))
    xy = np.full((n, len(side), 2), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            fid, players, b = line.rstrip("\n").split(":")
            k = int(fid) - 1
            v = np.array(players.replace(";", ",").split(","), np.float64).reshape(-1, 3)
            xy[k] = v[:, :2]
            ball[k, :2] = np.array(b.split(",")[:2], np.float64)
    # fractions (origin top-left, y down) -> metres from the centre spot, y up
    for arr in (xy, ball[:, None, :2]):
        arr[..., 0] = (arr[..., 0] - 0.5) * length
        arr[..., 1] = (0.5 - arr[..., 1]) * width
    period = np.zeros(n, np.int8)
    g = GAMES[game]
    for half, (start, end) in g["halves"].items():
        period[start - g["offsets"][half]:end - g["offsets"][half] + 1] = half
    return Match(
        name=game, source="barath-" + source, fps=float(GAMES[game]["fps"]),
        pitch=(length, width), home=xy[:, side == 0], away=xy[:, side == 1], ball=ball,
        period=period, alive=np.full(n, -1, np.int8), team=np.full(n, -1, np.int8),
        meta={"n_frames": n},
    )
