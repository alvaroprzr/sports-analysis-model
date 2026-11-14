Frozen E7 (fold models, teams-swapped inference averaged, submitted decoder) on held-out development sets. Criterion fixed beforehand: reproduced if D recall in `nogaps` < 0.8.

## DFL sim (held-out degradation of development)

| frames | D recall | D precision | D F1 | predicted D among live frames | dead frames | live frames |
|---|---|---|---|---|---|---|
| all | 0.980 | 0.962 | 0.971 | 0.033 | 330052 | 383437 |
| ball visible | 0.989 | 0.990 | 0.989 | 0.003 | 96964 | 295597 |
| ball missing | 0.976 | 0.951 | 0.963 | 0.134 | 233088 | 87840 |

- ball missing on 0.706 of the dead frames and 0.229 of the live frames; visible share 0.550; predicted D share 0.472 (true 0.463)
- alive/dead F1 (mean over matches): 0.974
- spellF1 (flag): 0.577  [0.532, 0.549, 0.608, 0.581, 0.617]
- spellF1 (control): 0.677  [0.636, 0.641, 0.724, 0.684, 0.699]

| stoppage duration | n | found | ball mostly visible: n, found | ball mostly missing: n, found |
|---|---|---|---|---|
| 0-5 s | 47 | 0.68 | 30, 0.77 | 17, 0.53 |
| 5-10 s | 77 | 0.92 | 34, 1.00 | 43, 0.86 |
| 10-20 s | 140 | 0.97 | 47, 1.00 | 93, 0.96 |
| 20-inf s | 256 | 1.00 | 54, 1.00 | 202, 1.00 |

## DFL clean (as provided)

| frames | D recall | D precision | D F1 | predicted D among live frames | dead frames | live frames |
|---|---|---|---|---|---|---|
| all | 0.999 | 0.969 | 0.984 | 0.028 | 330052 | 383437 |
| ball visible | 0.999 | 0.969 | 0.984 | 0.028 | 330052 | 383437 |
| ball missing | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 |

- ball missing on 0.000 of the dead frames and 0.000 of the live frames; visible share 1.000; predicted D share 0.478 (true 0.463)
- alive/dead F1 (mean over matches): 0.986
- spellF1 (flag): 0.674  [0.641, 0.634, 0.687, 0.724, 0.684]
- spellF1 (control): 0.748  [0.707, 0.741, 0.758, 0.739, 0.796]

| stoppage duration | n | found | ball mostly visible: n, found | ball mostly missing: n, found |
|---|---|---|---|---|
| 0-5 s | 47 | 0.89 | 47, 0.89 | 0, - |
| 5-10 s | 77 | 1.00 | 77, 1.00 | 0, - |
| 10-20 s | 140 | 1.00 | 140, 1.00 | 0, - |
| 20-inf s | 256 | 1.00 | 256, 1.00 | 0, - |

## SkillCorner, 17 development matches (approximate dead-ball label; spellF1 live)

| frames | D recall | D precision | D F1 | predicted D among live frames | dead frames | live frames |
|---|---|---|---|---|---|---|
| all | 0.280 | 0.985 | 0.437 | 0.004 | 464675 | 537704 |
| ball visible | 0.240 | 0.970 | 0.385 | 0.003 | 220735 | 528859 |
| ball missing | 0.317 | 0.996 | 0.481 | 0.036 | 243940 | 8845 |

- ball missing on 0.525 of the dead frames and 0.016 of the live frames; visible share 0.748; predicted D share 0.132 (true 0.464)
- alive/dead F1 (mean over matches): 0.760
- spellF1 (live): 0.810  [0.837, 0.809, 0.810, 0.786, 0.802, 0.826, 0.778, 0.767, 0.839, 0.800, 0.818, 0.835, 0.804, 0.791, 0.854, 0.828, 0.792]

| stoppage duration | n | found | ball mostly visible: n, found | ball mostly missing: n, found |
|---|---|---|---|---|
| 0-5 s | 517 | 0.19 | 493, 0.20 | 24, 0.08 |
| 5-10 s | 541 | 0.29 | 352, 0.32 | 189, 0.25 |
| 10-20 s | 520 | 0.28 | 175, 0.17 | 345, 0.34 |
| 20-inf s | 197 | 0.31 | 32, 0.19 | 165, 0.33 |

## DFL nogaps (degradation without its ball-gap step)

| frames | D recall | D precision | D F1 | predicted D among live frames | dead frames | live frames |
|---|---|---|---|---|---|---|
| all | 0.996 | 0.995 | 0.996 | 0.004 | 330052 | 383437 |
| ball visible | 0.996 | 0.995 | 0.996 | 0.004 | 330052 | 383437 |
| ball missing | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 |

- ball missing on 0.000 of the dead frames and 0.000 of the live frames; visible share 1.000; predicted D share 0.464 (true 0.463)
- alive/dead F1 (mean over matches): 0.996
- spellF1 (flag): 0.649  [0.602, 0.630, 0.670, 0.666, 0.679]
- spellF1 (control): 0.739  [0.715, 0.721, 0.766, 0.742, 0.749]

| stoppage duration | n | found | ball mostly visible: n, found | ball mostly missing: n, found |
|---|---|---|---|---|
| 0-5 s | 47 | 0.81 | 47, 0.81 | 0, - |
| 5-10 s | 77 | 1.00 | 77, 1.00 | 0, - |
| 10-20 s | 140 | 1.00 | 140, 1.00 | 0, - |
| 20-inf s | 256 | 1.00 | 256, 1.00 | 0, - |

## DFL shuffled (the gaps of `sim` at positions independent of the game)

| frames | D recall | D precision | D F1 | predicted D among live frames | dead frames | live frames |
|---|---|---|---|---|---|---|
| all | 0.976 | 0.928 | 0.951 | 0.065 | 330052 | 383437 |
| ball visible | 0.996 | 0.994 | 0.995 | 0.005 | 177479 | 215094 |
| ball missing | 0.951 | 0.859 | 0.903 | 0.142 | 152573 | 168343 |

- ball missing on 0.462 of the dead frames and 0.439 of the live frames; visible share 0.550; predicted D share 0.486 (true 0.463)
- alive/dead F1 (mean over matches): 0.956
- spellF1 (flag): 0.527  [0.507, 0.495, 0.522, 0.581, 0.530]
- spellF1 (control): 0.629  [0.614, 0.600, 0.630, 0.671, 0.630]

| stoppage duration | n | found | ball mostly visible: n, found | ball mostly missing: n, found |
|---|---|---|---|---|
| 0-5 s | 47 | 0.64 | 23, 0.83 | 24, 0.46 |
| 5-10 s | 77 | 0.90 | 45, 1.00 | 32, 0.75 |
| 10-20 s | 140 | 0.98 | 78, 1.00 | 62, 0.95 |
| 20-inf s | 256 | 1.00 | 132, 1.00 | 124, 1.00 |

Criterion: D recall in `nogaps` = 0.996 -> not reproduced (>= 0.8)

## P(alive) of the frozen E7 before decoding (10 Hz rows, out of fold)

| source | AUC of dead vs live | P(alive) on dead rows, percentiles 5 / 25 / 50 / 75 / 95 | on live rows | recall of D at P(alive) < 0.5 | < 0.9 | < 0.99 | predicted D among live at < 0.5 | < 0.9 | < 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| dfl_sim | 0.997 | 0.000, 0.001, 0.003, 0.024, 0.344 | 0.716, 0.992, 0.999, 0.999, 1.000 | 0.972 | 0.997 | 1.000 | 0.023 | 0.107 | 0.232 |
| dfl_clean | 1.000 | 0.000, 0.000, 0.000, 0.001, 0.008 | 0.922, 0.997, 0.999, 0.999, 1.000 | 0.999 | 1.000 | 1.000 | 0.026 | 0.047 | 0.099 |
| sc | 0.934 | 0.193, 0.543, 0.821, 0.968, 0.999 | 0.985, 0.998, 0.999, 0.999, 1.000 | 0.219 | 0.614 | 0.832 | 0.003 | 0.016 | 0.064 |

## SkillCorner by the number of players really detected (the others are extrapolated)

| players detected | share of frames | dead share | D recall | D precision | predicted D among live |
|---|---|---|---|---|---|
| 0 | 0.365 | 0.916 | 0.303 | 0.992 | 0.026 |
| 1-7 | 0.017 | 0.310 | 0.208 | 0.940 | 0.006 |
| 8-13 | 0.198 | 0.194 | 0.259 | 0.957 | 0.003 |
| 14-17 | 0.267 | 0.209 | 0.233 | 0.961 | 0.002 |
| 18-22 | 0.153 | 0.196 | 0.152 | 0.972 | 0.001 |

## The visible ball while dead and while live, by source (clean tracking as provided)

| source | state | frame-to-frame speed m/s, percentiles 10 / 25 / 50 / 75 / 90 | exactly 0 | < 0.5 m/s | a really observed player within 1.5 m |
|---|---|---|---|---|---|
| dfl | dead | 0.00, 0.00, 0.00, 0.00, 0.00 | 1.000 | 1.000 | 0.120 |
| dfl | live | 1.58, 3.34, 6.29, 11.31, 16.10 | 0.007 | 0.023 | 0.526 |
| skillcorner | dead | 0.30, 0.61, 1.30, 2.91, 5.81 | 0.008 | 0.195 | 0.183 |
| skillcorner | live | 1.56, 3.03, 5.50, 9.52, 14.15 | 0.000 | 0.017 | 0.463 |
