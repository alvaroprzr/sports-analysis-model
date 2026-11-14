# Ball possession, frame by frame — BAR-ATH

Our solution to Phase 1 of More than a Hack 2027: for every video frame of the BAR-ATH match, say
whether the home team (H) or the away team (A) has the ball, or the ball is dead (D). The challenge
statement, the rules, the data description and the FAQ are in the organisers' official repository:
<https://github.com/metrica-sports/more-than-a-hack-2027>. The documents this repository's code and text refer to by name
(CHALLENGE.md, DATA.md, FAQ.md, SUBMISSION.md, RULES.md, EVALUATION.md) are the ones there.

The submitted file combines two things: which team has the ball comes from E7, an equal-weight
combination of four sequence models (a set encoder per frame, a bidirectional GRU and a CRF at
5 Hz); whether the ball is in play comes from a gradient-boosting classifier trained on real
broadcast tracking and on DFL with the provider's motionless dead ball hidden. A three-state
Viterbi decoder and the ball-out rule follow. All models are trained, and all parameters and thresholds are tuned, only on external open data (IDSSE / DFL and
SkillCorner), with DFL tracking degraded by a simulator calibrated on SkillCorner. The match itself
is never used to train or tune anything. The method, the data and how it was built are in the
**[technical document](docs/TECHNICAL_DOCUMENT.md)**.

## Installation

Python 3.12, CPU only. Byte-identical regeneration is claimed for these exact versions:

    pip install -r requirements-solution.txt
    pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

## Data

- **The match bundle** (gated, not in this repository): enrol and download it as described in
  DATA.md of the official repository, then unpack it into `data/` ([data/README.md](data/README.md)) or point
  `MTAH_DATA_DIR` to it. This is all that regenerating the submission needs.
- **External data**, only to retrain the models or rebuild the reserve submission (B1):

      python scripts/fetch_dfl.py
      git clone https://github.com/SkillCorner/opendata.git data/external/skillcorner_raw/repo
      git -C data/external/skillcorner_raw/repo checkout 4340d274572876239c154c90bc507a9b3250a656
      python scripts/fetch_skillcorner.py --lfs && python scripts/fetch_skillcorner.py

  Sources, licences, versions and checksums (including the DFL event files from figshare) are in
  [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Regenerating the submission

From the saved models in `models/` (inference only, CPU):

    python scripts/make_submission_b2ens.py --team <team> --config configs/b2_ens_dr_alive.json
    python evaluation/validate_submission.py <team>.csv

The file has SHA-256

    236b8f603eccf74dcf08d3be22612cac997af8c8a66a066bc84dec19d845f7ac

The saved stoppage classifier (`models/alive/alive_g.txt`) is a text file that must reach the
reader without end-of-line conversion; `.gitattributes` guarantees it in a git checkout on any
platform.

With `--config configs/b2_ens_dr.json` the same script gives E7 alone, as frozen on 12 November
(SHA-256 df9bc4c947cc7999145c837434220ad7d6d1acab00c153f23c31362f5ecc5a93).

Retraining, the reserve submission and cross-machine determinism are covered in section 9 of the
[technical document](docs/TECHNICAL_DOCUMENT.md#9-reproduction).

## Layout

| folder | contents |
|---|---|
| `src/poss/` | features, simulator, models, decoder |
| `scripts/` | data download, development, selection and submission scripts |
| `configs/` | the submitted configuration (`b2_ens_dr_alive.json`), E7 alone (`b2_ens_dr.json`) and the reserve (B1) |
| `models/dr/`, `models/alive/` | the saved weights of the four E7 members; the saved stoppage classifier |
| `experiments/` | reports of every experiment and choice (aggregates only) |
| `docs/` | technical document, data sources, B1 |
| `evaluation/`, `starter/`, `games.py` | the organisers' evaluation and starter code and match constants, reused unchanged from the [official repository](https://github.com/metrica-sports/more-than-a-hack-2027) under its licence |

The organisers' licence and citation apply to their material ([LICENSE](LICENSE),
[CITATION.cff](CITATION.cff)).
