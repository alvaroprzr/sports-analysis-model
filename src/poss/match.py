"""Common in-memory format shared by every data source.

One `Match` holds one game at its native frame rate:

    home, away   (T, P, 2) float32  player positions in metres, NaN when the slot is not tracked
    ball         (T, 3)    float32  ball x, y, z in metres, NaN when the ball is not tracked
    period       (T,)      int8     1, 2, ... (0 = outside play, e.g. half-time)
    alive        (T,)      int8     1 alive, 0 dead, -1 unknown (unlabelled source)
    team         (T,)      int8     0 home, 1 away, -1 none / unknown

Coordinates: origin at the centre spot, x along the length, y along the width, both in metres.
Attacking direction is not normalised here; features decide what to do with it.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Match:
    name: str
    source: str
    fps: float
    pitch: tuple
    home: np.ndarray
    away: np.ndarray
    ball: np.ndarray
    period: np.ndarray
    alive: np.ndarray
    team: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.period)

    def save(self, path):
        np.savez_compressed(
            path, name=self.name, source=self.source, fps=self.fps, pitch=np.asarray(self.pitch),
            home=self.home, away=self.away, ball=self.ball, period=self.period,
            alive=self.alive, team=self.team,
            **{"meta_" + k: v for k, v in self.meta.items() if isinstance(v, np.ndarray)},
        )

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(
            name=str(z["name"]), source=str(z["source"]), fps=float(z["fps"]),
            pitch=tuple(float(v) for v in z["pitch"]), home=z["home"], away=z["away"],
            ball=z["ball"], period=z["period"], alive=z["alive"], team=z["team"],
            meta={k[5:]: z[k] for k in z.files if k.startswith("meta_")},
        )


def from_kloppy(ds, name, source, to_metres, label_states=True):
    """Convert a kloppy TrackingDataset into a `Match`.

    `to_metres(x, y)` maps the dataset's coordinates to the centred metric frame. Player slots are
    the team's roster order in the metadata, so substitutes get their own slot and are NaN while
    off the pitch.
    """
    home_team, away_team = ds.metadata.teams
    slots = {}
    for side, tm in enumerate((home_team, away_team)):
        for i, p in enumerate(tm.players):
            slots[p.player_id] = (side, i)
    n = len(ds.frames)
    home = np.full((n, len(home_team.players), 2), np.nan, np.float32)
    away = np.full((n, len(away_team.players), 2), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    period = np.zeros(n, np.int8)
    alive = np.full(n, -1, np.int8)
    team = np.full(n, -1, np.int8)
    for k, fr in enumerate(ds.frames):
        period[k] = fr.period.id
        b = fr.ball_coordinates
        if b is not None and b.x is not None and np.isfinite(b.x):
            ball[k, :2] = to_metres(b.x, b.y)
            ball[k, 2] = b.z if getattr(b, "z", None) is not None else np.nan
        for p, pt in fr.players_coordinates.items():
            side, i = slots[p.player_id]
            (home if side == 0 else away)[k, i] = to_metres(pt.x, pt.y)
        if label_states:
            if fr.ball_state is not None:
                alive[k] = int(fr.ball_state.value == "alive")
            if fr.ball_owning_team is not None:
                team[k] = 0 if fr.ball_owning_team.team_id == home_team.team_id else 1
    pitch = (ds.metadata.pitch_dimensions.pitch_length, ds.metadata.pitch_dimensions.pitch_width)
    return Match(name, source, float(ds.metadata.frame_rate), pitch, home, away, ball, period,
                 alive, team)
