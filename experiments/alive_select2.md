B', last attempt: arm g against the frozen E7 under the rule of scripts/alive_select2.py, fixed before g was trained. Five folds; held-out sets: 17 SkillCorner, 5 DFL with the dead ball hidden, and the same 5 DFL with the ball gaps at positions independent of the game (stress). Decoder chosen excluding the evaluated fold, without the stress copies. Arm c is shown for reference.

## Scores (spellF1)

| P(alive) | decoder per fold (c_team, c_stop) | SC full, metric 1 | DFL stress, flag | DFL stress, control | DFL hidden, flag | DFL hidden, control | SC live (does not decide) |
|---|---|---|---|---|---|---|---|
| E7 frozen | (3, 8), (3, 8), (3, 8), (3, 8), (3, 8) | 0.6514 | 0.5016 | 0.6123 | 0.5814 | 0.6903 | 0.8103 |
| c: both | (3, 4), (3, 4), (3, 6), (2, 6), (2, 6) | 0.7390 | 0.5075 | 0.6004 | 0.6181 | 0.7235 | 0.7873 |
| g: c + gaps independent of the game | (2, 12), (3, 2), (2, 12), (2, 12), (2, 12) | 0.7377 | 0.5188 | 0.6136 | 0.6230 | 0.7158 | 0.7854 |

## Predicted stoppages on SkillCorner (>= 2 s, >= 14 detected players on most frames)

| P(alive) | predicted stoppages per match | real (labelled dead or a restart event) | restart events with predicted D in the 5 s before |
|---|---|---|---|
| E7 frozen | 11.0 | 0.989 | 0.278 |
| c: both | 15.8 | 0.978 | 0.928 |
| g: c + gaps independent of the game | 15.2 | 0.985 | 0.921 |

## The rule: arm minus the frozen E7, paired by match (SE)

| P(alive) | 1. SC full, metric 1 (> 2 SE) | 2. real stoppages (>= 0.85) | 3. stress flag (>= -1 SE) | 3. stress control (>= -1 SE) | 4. hidden flag (>= -1 SE) | 4. hidden control (>= -1 SE) | SC live (does not decide) | passes |
|---|---|---|---|---|---|---|---|---|
| c: both | +0.0876 (0.0047) | 0.978 | +0.0059 (0.0055) | -0.0119 (0.0055) | +0.0368 (0.0101) | +0.0331 (0.0120) | -0.0230 (0.0030) | no |
| g: c + gaps independent of the game | +0.0863 (0.0045) | 0.985 | +0.0172 (0.0026) | +0.0013 (0.0046) | +0.0416 (0.0116) | +0.0254 (0.0090) | -0.0249 (0.0036) | YES |

**Decision by the rule: arm g ADOPTED**

(The row of c only applies the new conditions to the earlier arm for reference; c was not a candidate.)

## Stoppages, frame level, after decoding

| P(alive) | SC, >= 14 detected: D recall | D precision | predicted D among live | DFL stress: D recall | D precision | predicted D among live | DFL hidden: D recall | D precision | predicted D among live |
|---|---|---|---|---|---|---|---|---|---|
| E7 frozen | 0.205 | 0.964 | 0.002 | 0.959 | 0.912 | 0.079 | 0.940 | 0.982 | 0.015 |
| c: both | 0.877 | 0.908 | 0.023 | 0.975 | 0.880 | 0.115 | 0.986 | 0.981 | 0.016 |
| g: c + gaps independent of the game | 0.877 | 0.904 | 0.024 | 0.961 | 0.923 | 0.069 | 0.984 | 0.983 | 0.015 |

Decoder of g on all folds: (2, 12)

{"adopted": "g", "costs": [2, 12.0, 12.0]}
