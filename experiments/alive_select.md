B': replacing P(alive) only (scripts/alive_dev.py, scripts/alive_select.py). Five folds, held-out sets: 5 DFL (dead ball hidden, calibrated degradation) and 17 SkillCorner. Decoder chosen excluding the evaluated fold.

## Scores (spellF1)

| P(alive) | decoder per fold (c_team, c_stop) | SC full, metric 1 | SC live | DFL hidden, flag | DFL hidden, control | SC full without neutralising |
|---|---|---|---|---|---|---|
| E7 frozen | (3, 8), (3, 8), (3, 8), (3, 8), (3, 8) | 0.6514 | 0.8103 | 0.5814 | 0.6903 | 0.5307 |
| a: SkillCorner only | (2, 12), (2, 12), (2, 8), (2, 12), (2, 8) | 0.7396 | 0.7871 | 0.4860 | 0.5718 | 0.7203 |
| b: DFL only | (2, 12), (3, 6), (2, 6), (2, 12), (2, 12) | 0.6227 | 0.8030 | 0.6230 | 0.7180 | 0.5263 |
| c: both | (3, 4), (3, 4), (3, 6), (2, 6), (2, 6) | 0.7390 | 0.7873 | 0.6181 | 0.7235 | 0.7214 |
| d: E7 recalibrated | (2, 12), (3, 12), (2, 12), (3, 4), (2, 12) | 0.6836 | 0.7944 | 0.5715 | 0.6643 | 0.6356 |

## The rule: arm minus the frozen E7, paired by match (SE)

| P(alive) | 1. SC full, metric 1 (> 2 SE) | 2. SC live (>= -1 SE) | 3. DFL flag (>= -1 SE) | 3. DFL control (>= -1 SE) | passes |
|---|---|---|---|---|---|
| a: SkillCorner only | +0.0882 (0.0054) | -0.0232 (0.0033) | -0.0954 (0.0131) | -0.1186 (0.0140) | no |
| b: DFL only | -0.0287 (0.0024) | -0.0073 (0.0017) | +0.0417 (0.0120) | +0.0277 (0.0100) | no |
| c: both | +0.0876 (0.0047) | -0.0230 (0.0030) | +0.0368 (0.0101) | +0.0331 (0.0120) | no |
| d: E7 recalibrated | +0.0322 (0.0034) | -0.0159 (0.0031) | -0.0098 (0.0153) | -0.0261 (0.0127) | no |

**Decision by the rule: no arm adopted; the frozen E7 stays**

## Stoppages, frame level, after decoding

| P(alive) | SC, >= 14 detected: D recall | D precision | predicted D among live | SC, all frames: D recall | D precision | DFL hidden: D recall | D precision | predicted D among live |
|---|---|---|---|---|---|---|---|---|
| E7 frozen | 0.205 | 0.964 | 0.002 | 0.280 | 0.985 | 0.940 | 0.982 | 0.015 |
| a: SkillCorner only | 0.888 | 0.912 | 0.022 | 0.964 | 0.938 | 0.999 | 0.759 | 0.274 |
| b: DFL only | 0.079 | 0.967 | 0.001 | 0.574 | 0.966 | 0.985 | 0.983 | 0.015 |
| c: both | 0.877 | 0.908 | 0.023 | 0.958 | 0.940 | 0.986 | 0.981 | 0.016 |
| d: E7 recalibrated | 0.599 | 0.842 | 0.029 | 0.785 | 0.946 | 0.997 | 0.888 | 0.109 |

## P(alive) before decoding, 10 Hz rows (SkillCorner: rows with >= 14 detected players)

Cross-provider transfer: arm a never saw a DFL match, arm b never saw a SkillCorner match.

| P(alive) | SC: AUC | D recall at 0.5 | predicted D among live at 0.5 | DFL hidden: AUC | D recall at 0.5 | predicted D among live at 0.5 |
|---|---|---|---|---|---|---|
| E7 frozen | 0.866 | 0.168 | 0.001 | 0.997 | 0.925 | 0.011 |
| a: SkillCorner only | 0.981 | 0.876 | 0.018 | 0.961 | 0.999 | 0.261 |
| b: DFL only | 0.773 | 0.000 | 0.000 | 0.999 | 0.984 | 0.013 |
| c: both | 0.980 | 0.865 | 0.019 | 0.999 | 0.984 | 0.015 |
| d: E7 recalibrated | 0.865 | 0.581 | 0.029 | 0.996 | 0.997 | 0.099 |

## Arm d: recalibration fitted without the evaluated fold (temperature, bias)

fold 0: (0.832, -3.048), fold 1: (0.870, -3.361), fold 2: (0.844, -3.133), fold 3: (0.830, -3.077), fold 4: (0.842, -3.036); all 17 matches: (0.8424, -3.1233)

Decoder on all folds per arm: a: (2, 12), b: (2, 12), c: (2, 6), d: (3, 6)

{"adopted": null}
