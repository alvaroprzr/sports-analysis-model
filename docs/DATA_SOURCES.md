# Data sources and how they are made comparable

Labelled data used: IDSSE / Sportec open DFL data and SkillCorner open data. No labels of the
provided match (BAR-ATH) are used to train, tune or choose anything.

## What each source has

| | BAR-ATH (provided) | DFL / IDSSE | SkillCorner open data |
|---|---|---|---|
| Licence / citation | challenge data | CC-BY 4.0; Bassek et al. 2025, *Sci. Data* 12:195 | MIT; github.com/SkillCorner/opendata @ 4340d27 |
| Matches | 1 | 7 (dev 5, test 2) | 20 (dev 17, test 3) |
| Capture | Metrica automatic tracking, tactical camera, uncorrected | multi-camera optical, provider-corrected | broadcast video, players outside the view extrapolated |
| Native rate | 25 fps | 25 fps | 10 fps |
| Ball x, y | yes, missing on 41 % of played frames | always present (also when dead) | missing on 26 % (mostly when dead) |
| Ball height | no | yes | yes |
| Speeds given | yes (players; ball mostly NaN) | no | no |
| Stable identities | no: tracks fragment (median 35 s), 42 jumps > 12.4 m/s per player-hour | yes (one track per player per match) | yes, but 43 % of positions are extrapolated |
| Goalkeeper flag | yes (metadata `position_type`) | yes (roster position) | yes (`player_role`) |
| Per-frame labels | none used | ball status + owning team, from the provider | team in possession from SkillCorner's phases of play (model output, accuracy not published); dead ball only as gaps between phases |
| Possession granularity | unknown | 22.8 per 5 min, median 3.8 s | 12.9 per 5 min, median 7.7 s |

SkillCorner versions (commit 4340d27, 2026-09-14; MIT licence for the whole repository): 20 A-League
2024/25 matches played 2024-11-23 to 2025-05-17. The repository README still says 10: ten matches
were added on 2026-09-08 (commit abbc3ff: 1874553, 1927964, 1959846, 1986691, 1996436, 2006363,
2007448, 2007721, 2010085, 2016236). Dynamic events and phases of play are version 3 for all 20
(six refreshed in abbc3ff, four in 4340d27, the ten new ones added as v3). Tracking files are
the original LFS objects, verified against their sha256.

DFL event files (same IDSSE dataset, CC-BY 4.0, figshare article 28196177, DOI
10.6084/m9.figshare.28196177.v1): the seven `DFL_03_02_events_raw_*.xml` and
`DFL_02_01_matchinformation_*.xml` files, downloaded directly from figshare on 2026-11-05 into
`data/external/dfl_events_raw/`, each verified against the MD5 that figshare publishes (all 14
match; list in that folder's `manifest.txt`). kloppy's `load_open_event_data` reads a mirror and
does not verify checksums, so it was not used for the download. **Use: only to derive training and
evaluation labels of the DFL matches (whether a change of the owning team is a controlled
possession); never as a model input.**

Figures: `experiments/domain_stats.md`, `experiments/dataset_audit.md` (aggregates only).

## How they are homogenised

| Difference | Rule |
|---|---|
| Inputs | Only what the provided tracking has: player x, y, team, ball x, y when visible. Ball height, provider speeds and the goalkeeper flag are never inputs. |
| Derived quantities | Velocities, accelerations, distances, ball regime are computed by the same code from positions in every source, BAR-ATH included. |
| Frame rate | Physics is computed at the native rate and aggregated to 10 Hz rows (common rate). B1 works on the 10 Hz rows; B2 and the submitted combination (E7) take every second row (5 Hz). Sampling the DFL labels at 10 Hz and holding them back to 25 fps costs 0.005 spellF1 (0.011 at 5 Hz). |
| Coordinates | Metres, origin at the centre spot, x along the length, y along the width. |
| Missing ball / players | Explicit observation masks. DFL is degraded by a simulator calibrated on SkillCorner tracking (ball gaps, track cuts, player dropout; src/poss/sim_external.py), each training copy at its own intensity (domain randomisation). Nothing of BAR-ATH is used to calibrate it. |
| SkillCorner extrapolation | E7 and B1 are trained and scored with SkillCorner's player positions as provided, the extrapolated ones included. The simulator calibration (src/poss/sim_external.py) and the stoppage classifier of the submission (scripts/alive_dev.py) treat the players SkillCorner did not detect as missing; the classifier uses only the rows with at least 14 detected players. |
| Labels | DFL is the main reference. SkillCorner labels are a noisy reference with lower weight; E7 and B1 do not use its dead-ball label in any loss (team head on live frames only) because it is only approximate. The stoppage classifier of the submission is trained on it (rows with at least 14 detected players, half the total weight of DFL) and scored against it; the restart events of SkillCorner's dynamic events (`game_interruption_before`) are used for one check of its predicted stoppages, never as an input. |
