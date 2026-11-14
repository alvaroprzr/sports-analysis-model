Predictions: f4d_swap (out of fold, 22 held-out sets: 5 DFL sim, 17 SkillCorner). Minimum duration not adopted.

## 1. Decoder (c_team, c_stop, minimum duration k)

Excluding the evaluated fold, (c_team, c_stop, k) per fold: [(3, 8.0, 1), (3, 8.0, 1), (3, 8.0, 1), (3, 8.0, 1), (3, 2.0, 1)]

| decoder | reading | DFL sim | SC live | combined |
|---|---|---|---|---|
| chosen excluding the fold | flag | 0.5758 | 0.8097 | 0.6538 |
| chosen excluding the fold | control | 0.6755 | 0.8097 | 0.7202 |

For the submission (all folds): c_team 3, c_stop 8, k 1 rows.

## 2. Ball out of the pitch (F5, law of the game)

Reference c_team per fold [3, 3, 3, 3, 3]; candidate (c_team, (min_out_m, min_s)) per fold [(3, (2.0, 0.3)), (3, (2.0, 0.3)), (3, (2.0, 0.3)), (3, (2.0, 0.3)), (3, (1.0, 0.3))]

Detection check at min_out_m 2, min_s 0.3: share of the marked rows that are truly stopped (rows, share)

| source | all | goal line | touchline | after a gap (<= 1 s) | no gap | worst match |
|---|---|---|---|---|---|---|
| DFL sim | 1149, 0.99 | 583, 0.99 | 566, 0.99 | 95, 1.00 | 1054, 0.99 | 0.97 (109 rows) |
| SC live | 23959, 0.97 | 9075, 0.97 | 14884, 0.97 | 209, 0.94 | 23750, 0.97 | 0.89 (971 rows) |

No systematic error found (every cell with >= 50 rows at 0.5 or more).

| decoder | reading | DFL sim | SC live | combined |
|---|---|---|---|---|
| reference | flag | 0.5772 | 0.8093 | 0.6546 |
| reference | control | 0.6769 | 0.8093 | 0.7211 |
| candidate | flag | 0.5772 | 0.8106 | 0.6550 |
| candidate | control | 0.6769 | 0.8106 | 0.7215 |

Combined, mean of the two readings: +0.0004 (paired SE 0.0002)
- DFL sim, flag: +0.0000 (SE 0.0000)
- SC live, flag: +0.0013 (SE 0.0005)
- DFL sim, control: +0.0000 (SE 0.0000)
- SC live, control: +0.0013 (SE 0.0005)

Rule (law of the game: applied unless a source is worse than 2 SE): ADOPTED

For the submission: c_team 3, min_out_m 2, min_s 0.3.

## 3. Boundary shifts (heuristic)

Mean signed offset of the boundaries, predicted minus true (s; matched within 2 s), decoder so far at its submission parameters:

| source | reading | team (n, mean, median) | d_start (n, mean, median) | d_end (n, mean, median) |
|---|---|---|---|---|
| DFL sim | flag | 914, -0.02, +0.04 | 448, -0.03, +0.00 | 418, +0.13, +0.08 |
| DFL sim | control | 608, +0.01, +0.08 | 448, -0.03, +0.00 | 418, +0.13, +0.08 |
| SC live | flag | 1911, +0.08, +0.10 | 508, +0.70, +0.80 | 379, -0.36, -0.20 |

(SkillCorner labels have no stoppages of their own beyond out of play; its live score uses only the team boundaries.)

Candidate (c_team, shifts ('team', 'd_start', 'd_end') in s) per fold: [(3, (0.0, 0.0, -0.2)), (3, (0.0, 0.0, 0.0)), (3, (0.0, 0.0, 0.0)), (3, (0.0, 0.0, -0.2)), (2, (0.0, 0.0, -0.2))]

| decoder | reading | DFL sim | SC live | combined |
|---|---|---|---|---|
| reference | flag | 0.5773 | 0.8103 | 0.6550 |
| reference | control | 0.6769 | 0.8103 | 0.7214 |
| candidate | flag | 0.5790 | 0.8093 | 0.6557 |
| candidate | control | 0.6737 | 0.8093 | 0.7189 |

Combined, mean of the two readings: -0.0009 (paired SE 0.0006)
- DFL sim, flag: +0.0016 (SE 0.0021)
- SC live, flag: -0.0010 (SE 0.0010)
- DFL sim, control: -0.0032 (SE 0.0032)
- SC live, control: -0.0010 (SE 0.0010)

Rule (strict: combined > 2 SE, no source worse than 1 SE): not adopted

## Decoder of the submission

{"costs": [3, 8.0, 8.0], "min_possession_rows": 1, "ball_out": {"min_out_m": 2.0, "min_s": 0.3}, "boundary_shift_s": null}
