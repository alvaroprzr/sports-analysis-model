Exploratory, after scripts/alive_select.py gave its answer (no arm adopted); nothing is chosen with this.
Arm minus the frozen E7 at fixed decoder costs, paired by match (SE): the loss of the arms on SkillCorner
live does not come from the decoder costs (scripts/alive_posthoc.py).

| P(alive), decoder (c_team, c_stop) | SC full, metric 1 | SC live | DFL hidden, flag | DFL hidden, control | SC full without neutralising |
|---|---|---|---|---|---|
| c (3, 8) | +0.0894 (0.0050) | -0.0227 (0.0032) | +0.0346 (0.0113) | +0.0366 (0.0102) | +0.1936 (0.0076) |
| c (3, 6) | +0.0892 (0.0051) | -0.0218 (0.0031) | +0.0350 (0.0111) | +0.0372 (0.0099) | +0.1929 (0.0076) |
| c (3, 12) | +0.0883 (0.0053) | -0.0250 (0.0034) | +0.0335 (0.0107) | +0.0356 (0.0096) | +0.1933 (0.0079) |
| c (2, 6) | +0.0859 (0.0048) | -0.0237 (0.0034) | +0.0448 (0.0132) | +0.0273 (0.0100) | +0.1880 (0.0070) |
| a (3, 8) | +0.0938 (0.0055) | -0.0185 (0.0032) | -0.1031 (0.0104) | -0.1151 (0.0116) | +0.1954 (0.0080) |
| d (3, 8) | +0.0334 (0.0036) | -0.0116 (0.0021) | -0.0131 (0.0129) | -0.0187 (0.0127) | +0.1051 (0.0066) |
| frozen E7 (2, 6) | -0.0046 (0.0015) | -0.0055 (0.0015) | +0.0124 (0.0027) | -0.0075 (0.0036) | -0.0018 (0.0017) |
