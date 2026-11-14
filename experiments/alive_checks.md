Checks T1 and T2 of arm c of B', fixed on 13 November before measuring (scripts/alive_checks.py).

## T1. Predicted stoppages on SkillCorner (>= 2 s, >= 14 detected players on most frames)

| P(alive) | predicted stoppages per match | real (labelled dead or a restart event) | restart events with predicted D in the 5 s before |
|---|---|---|---|
| E7 frozen | 11.0 | 0.989 | 0.278 (84.5 per match) |
| c: both | 15.8 | 0.978 | 0.928 (84.5 per match) |

T1: 0.978 of the predicted stoppages of c are real -> PASS (>= 0.85)

## T2. DFL with ball gaps independent of the game and the dead ball hidden

Live rows without the ball: 0.440 (mean of 5 matches).

| P(alive) | spellF1 flag | spellF1 control | D recall | D precision | predicted D among live |
|---|---|---|---|---|---|
| E7 frozen | 0.5016 | 0.6123 | 0.959 | 0.912 | 0.079 |
| c: both | 0.5075 | 0.6004 | 0.975 | 0.880 | 0.115 |

T2: c minus frozen, flag +0.0059 (SE 0.0055), control -0.0119 (SE 0.0055) -> FAIL

**Not both pass: arm c is not adopted; the frozen E7 stays**
