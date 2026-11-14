| case | ours | expected | |
|---|---|---|---|
| perfect prediction | 1.0000 | 1.0000 | ok |
| X inside a possession (filled with previous label) | 1.0000 | 1.0000 | ok |
| labels on ungraded frames ignored | 1.0000 | 1.0000 | ok |
| partial overlap equals the official scorer | 0.9333 | 0.9333 | ok |
| one-to-one matching equals the official scorer | 0.5333 | 0.5333 | ok |
| 200 random sequences: max |ours - official| | 0.0000 | 0.0000 | ok |
| live score ignores dead-ball frames | 1.0000 | 1.0000 | ok |

7 of 7 checks pass. Graded-half selection on BAR-ATH: done by the organisers' scorer from the ground truth (frames outside the halves are X there); on our development matches frames outside play are X in our ground truth, so the same official functions skip them.
