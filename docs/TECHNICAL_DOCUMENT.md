# Frame-level possession from tracking: a model trained on external matches through a tracking simulator

## 0. Compliance in one paragraph

All models are trained, and all parameters and thresholds are tuned, on external labelled data
only (IDSSE / DFL and SkillCorner open data); no label or human-produced data of the provided game
is used by the method. The provided match (BAR-ATH) is read at prediction
time, as the input of the fixed pipeline, and by label-free automatic checks of the data and of the
output (listed in section 7). The tracking simulator is calibrated on SkillCorner tracking.

The submission is E7 with one part replaced after E7 was first frozen (12 November). A check on
external data (13 November) found that E7's stoppage probability does not transfer from DFL to
real automatic tracking, and why. P(alive) was then replaced by a classifier trained, selected and
adopted with external data only, under a rule fixed before it was trained (sections 3 and 6).

## 1. Approach

The provided match has no labels we may learn from, so the model learns possession from **other
matches that have per-frame labels**, and is made robust to **the defects of real, automatic
tracking** by training on labelled clean tracking that a simulator degrades. Three parts:

1. **A tracking simulator** (src/poss/simulate.py, src/poss/sim_external.py) turns clean, labelled
   DFL tracking into tracking with the defects of uncorrected automatic tracking: the ball is
   low-pass filtered and goes missing in stretches (hazard models of gap onset and length, which
   depend on what the ball is doing), player tracks are cut and restart, positions snap in rare
   corrections. Its statistics are calibrated on **three SkillCorner matches** (real broadcast
   tracking with its own defects: the ball is missing on 24 % of their frames, players leave the
   camera view), upsampled to 25 fps, with the players SkillCorner did not detect treated as
   missing. Rather than a single intensity, each training copy of a DFL match draws its own
   (**domain randomisation**): the share of visible ball uniformly between 20 % and 100 % (from
   clean to very degraded) and the track-cut hazard between none and three times the calibrated
   one. This arm was chosen against single intensities by a rule fixed beforehand (section 5).
2. **Player-ball physics** computed by the same code on every source, at the native frame rate,
   then aggregated to 10 Hz: who explains each change of the ball's velocity (distance, standing
   behind the ball with respect to its outgoing direction, speed along it), everyone near the
   ball (counts within 1 / 2 / 5 m, duels, first arrival along the ball path), the ball regime
   (flight / touch / control) and, inside ball gaps, the likely receiver and kicker. Features are
   team aggregates, so no player identity has to survive across frames.
3. **Probabilities and a segmenter**: P(alive) and P(home | alive) per frame, decoded by a
   three-state chain (H / A / D) whose costs set how often possession changes. Two models produce
   the probabilities:
   - **B1** (reserve): gradient boosting on the physics features, one row per 10 Hz frame.
   - **E7**: the equal-weight mean of four variants of a per-player model (set
     encoder + bidirectional GRU + linear-chain CRF, 5 Hz; section 3).

   **Submitted:** P(home | alive) of E7 and P(alive) of a stoppage classifier (gradient boosting
   on the physics features) trained on real broadcast tracking and on DFL with the provider's
   motionless dead ball hidden (section 3).

Literature that shaped the design: PathCRF (Kim et al., KDD 2026) and Ball Radar (Kim et al., KDD
2023) for player-set encoders and structured decoding, TranSPORTmer (Capellera et al., ACCV 2024)
for masked missing data, Lang et al. (Sci. Rep. 2022) for ball-in-play from players with long
temporal context, Tobin et al. (IROS 2017) for domain randomisation: training across a range of
simulator intensities instead of matching one target.

## 2. Assets and data

| asset of the challenge | used | how |
|---|---|---|
| tactical ATD (players, ball) | yes | the only input at prediction time |
| tactical metadata, games.py | yes | teams, pitch size, half boundaries |
| FAQ (frames 110808-113741 are a stoppage) | yes | those frames are written as D |
| broadcast ATD, homography, video | no | not needed (section 8: future work) |
| Smart Tagging | check only | a label-free check that the output's second half is aligned like the first (section 7); not an input (section 8) |

Every labelled dataset used, with its source, who labelled it and how much of it is used. All of
it is external; none comes from the provided game.

| source | who labelled it | licence | matches | how much is used |
|---|---|---|---|---|
| IDSSE / DFL open tracking (Bassek et al. 2025, Sci. Data 12:195) | the league's official data provider: positions from optical tracking, ball status (alive / dead) and owning team per frame recorded by the provider | CC BY 4.0 | 7 Bundesliga matches | 5 for training and every choice (5-fold cross-validation, one match per fold); 2 held out, never trained on or used to choose; 2 of the 5 (J03WMX, J03WN1) are the clean references of the simulator calibration |
| IDSSE event data, same dataset (figshare 10.6084/m9.figshare.28196177.v1, MD5-verified) | events logged manually by the provider's operators (passes, shots, ball claims, set pieces, duels) | CC BY 4.0 | the same 7 | labels only, never an input: a second reading of the DFL owning team (section 4), used to evaluate and choose, not to train |
| SkillCorner open data (github.com/SkillCorner/opendata, commit 4340d27) | automatic broadcast tracking; team in possession from SkillCorner's phases of play (the output of its own algorithms, accuracy not published); dead ball as the gaps between phases (approximate); restart events of its dynamic events, for one check only | MIT | 20 A-League matches | 17 for training (team head of E7 and B1, half weight; stoppage classifier, rows with at least 14 detected players, half weight) and every choice; 3 of those 17 calibrate the simulator and are excluded from choosing it; 3 held out, never used |

Data are not redistributed: scripts/fetch_dfl.py and scripts/fetch_skillcorner.py download them
from their sources (docs/DATA_SOURCES.md has versions and checksums).

## 3. The models

**B1.** LightGBM, two classifiers on the 48 physics features of each 10 Hz row: P(alive) trained
on simulated DFL (two degraded copies of each training match, teams also swapped), P(home | alive)
on simulated DFL and SkillCorner (SkillCorner rows weigh half in total).

**B2 and E7.** At 5 Hz each frame is a set of up to 11 player tokens per team (position in the
attacking frame, velocity, distance to the ball, whether the player stands behind the ball with
respect to its outgoing direction and their speed along it, ball-minus-player speed, a present
flag), a ball token (position, velocity, visible flag, impulses, distance beyond a line, position
inside a gap and its length) and the 48 physics features of B1. Eleven slots per team is the most
players of one team tracked at once anywhere in the external data (clean DFL, simulated DFL,
SkillCorner; experiments/max_per_team.md), which is also the number allowed on the pitch. Players
are ordered by distance to a reference point (the ball when visible, otherwise its last known
position, otherwise the centroid of the players; ties broken by slot, so the order is the same on
every CPU), so no identity has to persist. A set encoder (4-head attention over all players,
d = 48) pools each frame into a vector; a two-layer bidirectional GRU (64 units) reads 60 s
windows; a linear-chain CRF over H / A / D is trained jointly with two auxiliary per-frame heads,
P(alive) and P(home | alive), whose outputs feed the same decoder as B1. Training mixes simulated
DFL and SkillCorner windows (one third of each batch, half the weight of DFL, team head only),
AdamW (lr 2e-3, 100 warm-up steps), early stopping every 100 steps on held-out windows of the
training matches (patience 4, at most 3000 steps). Each final model is trained on all development
matches for the mean number of steps its folds chose.

E7 averages, with equal weights, P(alive) and P(home | alive) of four variants:

| member | change from E0 |
|---|---|
| E0 | attention over all players + GRU |
| E1 | attention over the 8 players nearest to the ball |
| E2 | mean / max pooling of player tokens instead of attention |
| E5 | auxiliary head "possession changes within 1 s" |

The architecture, its hyperparameters and the membership of the combination were chosen in a
first round of development, one factor at a time with the same tuning budget for every variant.
With the simulator calibrated on SkillCorner the four members were retrained and re-scored. On the out-of-fold
predictions of the final models, with E7's decoder, the combination still beats each member
(experiments/b2_e7_final.md):

| model | DFL simulated, flag | DFL simulated, second reading | SkillCorner live | E7 minus it, combined (paired SE) |
|---|---|---|---|---|
| E7 | 0.577 | 0.677 | 0.810 | |
| E0 | 0.563 | 0.653 | 0.803 | +0.015 (0.005) |
| E1 | 0.559 | 0.650 | 0.796 | +0.020 (0.003) |
| E2 | 0.556 | 0.648 | 0.798 | +0.021 (0.006) |
| E5 | 0.565 | 0.666 | 0.801 | +0.011 (0.005) |

**The stoppage classifier (P(alive) of the submission).** E7's P(alive) is learnt on simulated DFL
only. A check on external data made after E7 was frozen (experiments/gap_structure_check.md) showed
that it finds the stoppages of DFL (recall of D 0.98) but not those of real automatic tracking:
0.28 on SkillCorner against its approximate dead-ball label, with precision 0.99; P(alive) still
ranks the frames there (AUC 0.93), its level does not transfer. The cause is in the DFL data:
while the ball is dead the provider keeps it at exactly the same coordinates (frame-to-frame
displacement exactly 0 on 100.0 % of the dead frames, 0.7 % of the live ones), so on DFL a
motionless ball marks the stoppage. In automatic tracking the ball of a stoppage moves
(SkillCorner: median 1.3 m/s when visible) and that cue does not exist.

The submission therefore takes P(alive), and only P(alive), from a separate classifier
(scripts/alive_dev.py): LightGBM with the parameters of B1 on the 10 Hz rows of B1 without the
columns that count tracked players (how many players are tracked depends on the camera, not on the
game), averaged over the match and the match with the teams exchanged. It is trained on:
- **SkillCorner**, real broadcast tracking with its approximate dead-ball label: only the rows
  with at least 14 really detected players, with the players SkillCorner did not detect masked as
  missing (its extrapolation is not an observation, and 72 % of its dead frames have no detected
  player at all); half the total weight of DFL;
- **DFL with the ball hidden while it is dead**, hidden before the simulator degrades the match,
  in four copies per match: two with the simulator's ball gaps and two with the same gaps moved to
  positions independent of the game (the visible and the missing runs of each half permuted), so
  that a missing ball alone does not mean a stoppage (across those copies the ball is missing on
  5 % to 80 % of live play).

P(home | alive), the team-swapped inference, the ball-out rule and the rounding are E7's. How the
classifier was selected and what it changes is in section 6.

## 4. Possession rules implemented

The rules are those of the labelled matches, learnt rather than written by hand, plus a few
explicit decoder rules that follow the laws of the game.

- **Live and dead ball.** A frame is D when the DFL ball status says the ball is out of play
  (from the whistle or the ball leaving the pitch to the restart). In SkillCorner, which the
  stoppage classifier also learns from, a frame is D in the gaps between its phases of play, an
  approximate label. The frames 110808-113741 of the
  provided match are a stoppage by the FAQ and are written as D.
- **Owning team.** A live frame belongs to the team the DFL provider names as owning the ball. In
  that convention a team keeps possession until the other team controls the ball: a deflection, a
  block or a ball that only touches an opponent does not change possession by itself.
- **Loose balls and challenges (second reading).** The provider's flag sometimes switches teams on
  an uncontrolled touch. A second reading of the same labels, fixed before any model was scored,
  accepts a change of owner only if within 1.5 s the new team has a control event (pass, cross,
  shot, ball claim, set piece, a ground duel won with the ball kept), or keeps the ball for at
  least 3 s or three passes; otherwise the frames return to the previous owner. Models are trained
  on the provider's flag; both readings are used to evaluate and choose, and a change is adopted
  only if it holds under both.
- **Restarts.** Possession at a restart belongs to the team that takes it, from the frame the ball
  is back in play; the stoppage before it is D. Entering or leaving D costs c_stop nats in the
  decoder: 12 in the submission and 8 in E7 with its own P(alive), each chosen on its own
  probabilities (below); B1, the reserve, uses 6.
- **Granularity.** A change of possession between teams costs c_team nats, which removes
  sub-second flickers. The two labelled sources define possession at different granularities (DFL
  4.6 possessions per minute of play, median 3.8 s; SkillCorner 2.6, median 7.7 s), so c_team is
  taken where all sources are near their best at once (zone score), not at the optimum of one.
- **Decoder choices**, all made on the out-of-fold predictions of the final models'
  folds, with a policy fixed before those predictions were scored (scripts/decoder_select.py,
  experiments/decoder_select.md). Every parameter is chosen by the zone score over three sources
  (simulated DFL with the flag, with the second reading, SkillCorner live) excluding the evaluated
  fold, and each step is compared with the decoder as adopted so far:
  - team-swapped inference averaged (the model is applied to the match with the teams exchanged and
    the two answers averaged) and a minimum possession duration (experiments/f4d_rule.md): adopted
    only under the usual rule (combined score averaged over the two readings better by more than 2
    paired SE, no source worse by more than 1 paired SE in either reading). Team swap adopted
    (+0.0149, paired SE 0.0023, better in both sources and both readings); minimum duration not
    (+0.0018, SE 0.0013), nor overlapping 60 s windows (+0.0027, SE 0.0015);
  - c_team 3 (the same in all five folds when each is excluded), c_stop 8, no minimum duration;
  - **ball out (law of the game: a ball wholly over a line is out of play).** The visible ball more
    than min_out_m beyond a line for at least min_s and not in fast flight (projection of a high
    ball) gives P(alive) ~ 0 before decoding. This is a rule of the laws of the game with an
    imperfect computational detection, not a statistically demonstrated improvement: it is applied
    with thresholds chosen on development unless some source gets worse by more than 2 paired SE
    (a guard against implementation errors). Its measured effect on development is within noise for
    the usual rule (combined +0.0004, paired SE 0.0002): it is in the submission through this
    policy for rules of the laws of the game, not through the usual rule. Origin of the idea: a
    label-free automatic check of an output on BAR-ATH (visible-ball episodes outside the
    lines with no predicted stoppage nearby; scripts/dead_ball_check.py); the rule,
    its thresholds and its adoption use external data only. The detection was checked for systematic false
    outs in both sources (share of the marked rows truly stopped, by line, after a ball gap, per
    match). Applied with min_out_m 2.0 m and min_s 0.3 s: SkillCorner +0.0013 (SE 0.0005), DFL
    unchanged (its ball-out stretches were already decoded as stoppages); of the rows it marks, 99 %
    (DFL) and 97 % (SkillCorner) are truly stopped, alike at goal lines and touchlines and after a
    ball gap, 89 % in the worst match;
  - boundary shifts (every boundary of one type, team change, start or end of a stoppage, moved by
    +-0.2 or +-0.4 s before the 25 fps output): a heuristic, so only under the usual rule; not
    adopted (combined -0.0009, SE 0.0006). The measured offsets on DFL were already small (mean
    -0.03 to +0.13 s per type);
  - **with the stoppage classifier** (section 3) the two costs were chosen again by the same zone
    score, excluding the evaluated fold, over DFL with the dead ball hidden (flag and second
    reading) and SkillCorner over the whole match: c_team 2 and c_stop 12, the choice of four of
    the five folds and of all folds together (experiments/alive_select2.md). Team swap, ball-out
    rule and rounding are unchanged.

## 5. How the simulator was chosen (external data only)

Six versions of B1 were trained on five folds, differing only in how DFL was degraded: clean (no
simulator), the simulator as calibrated on SkillCorner, twice and half the ball-gap onset, twice the
track cuts, and domain randomisation. They were scored on the 14 SkillCorner development matches not
used for the calibration, at three levels fixed beforehand: as they are, and with extra ball gaps
until 70 % and 40 % of the ball is visible. Rule, fixed before the results: the best mean of the
three levels wins, provided it is not more than 1 paired SE below the best arm on SkillCorner as it
is (experiments/sim_ext_select2.md).

| arm | SkillCorner as is | 70 % ball | 40 % ball | mean of 3 | clean DFL (second reading; not in the rule) |
|---|---|---|---|---|---|
| clean DFL | 0.767 | 0.754 | 0.664 | 0.728 | 0.678 |
| as calibrated | 0.770 | 0.768 | 0.724 | 0.754 | 0.569 |
| ball gaps x2 | 0.773 | 0.770 | 0.731 | 0.758 | 0.553 |
| ball gaps x0.5 | 0.771 | 0.765 | 0.719 | 0.752 | 0.568 |
| track cuts x2 | 0.777 | 0.770 | 0.733 | 0.760 | 0.554 |
| **domain randomisation** | **0.780** | **0.772** | 0.732 | **0.761** | 0.547 |

Domain randomisation wins and is also the best on SkillCorner as it is; on SkillCorner every
simulated arm beats clean training, most where the ball is often missing (+0.068 at 40 % visible
ball). On clean DFL tracking the opposite holds for B1: every simulated arm loses about 0.11-0.13
to clean training, the chosen arm most. The rule scores real automatic tracking, the kind of
tracking of the provided match, so the arm was kept. E7 does not show that loss: its fold models
score 0.748 on clean DFL (second reading) against 0.677 on simulated DFL
(experiments/gap_structure_check.md). The SkillCorner
matches are never degraded by the simulator for training, so this is an independent check that the
simulator moves the model towards real tracking defects rather than towards its own artefacts.

## 6. Results on held-out development matches (5-fold cross-validation)

Scores: spellF1 on simulated DFL (whole match) and on SkillCorner (live play), costs chosen
excluding the evaluated fold.

| model | DFL simulated, flag | DFL simulated, second reading | SkillCorner live |
|---|---|---|---|
| B1 (reserve), c_team 3 | 0.480 | 0.630 | 0.789 |
| E7, c_team 3, c_stop 8, team swap, ball-out rule | 0.577 | 0.677 | 0.811 |

These development figures are optimistic: the 14 SkillCorner matches that chose the simulator
(section 5) are also among those that score E7, and the decoder was chosen on the same folds (each
fold excluded from its own choice, but the models of the other folds saw it). The clean estimate is
the reserved test.

**Reserved test** (2 DFL, 3 SkillCorner; never used to train, choose or tune; queried once, after
everything else was fixed, with the fold models of the four members of E7 averaged and E7's
decoder; nothing was changed after it; experiments/test_once.md): DFL simulated 0.624 with the
flag and 0.721 with the second reading, SkillCorner live 0.814 (per match 0.838, 0.783, 0.819). On
SkillCorner, the clean test, the score matches the development estimate. The two DFL test matches
had been seen during earlier development; the SkillCorner test is the clean one. The
test was not queried again for the stoppage classifier (below).

**What the SkillCorner figure does and does not measure.** It is the spellF1 of possessions inside
live play: frames SkillCorner marks as stopped are not graded and a predicted stoppage there counts
as an abstention (src/poss/evaluate.py), because its dead-ball label is only approximate. It
therefore measures the assignment of the team on real automatic tracking, not the detection of
stoppages. E7's P(alive) is trained on simulated DFL only and its scores come from DFL only; that
is the gap the stoppage classifier closes.

**The stoppage classifier against E7** (experiments/alive_select2.md). Five folds; the classifier
of each fold is trained without its matches and its decoder costs are chosen excluding the fold;
the reference is E7 with its own P(alive) and decoder on the same sets. On DFL the ball is hidden
while dead for both.

| held-out set, spellF1 over the whole match | E7 | submitted | difference (paired SE) |
|---|---|---|---|
| SkillCorner, 17 matches | 0.651 | 0.738 | +0.086 (0.005) |
| DFL, simulator's ball gaps, flag | 0.581 | 0.623 | +0.042 (0.012) |
| DFL, simulator's ball gaps, second reading | 0.690 | 0.716 | +0.025 (0.009) |
| DFL, gaps independent of the game (44 % of live play without the ball), flag | 0.502 | 0.519 | +0.017 (0.003) |
| DFL, gaps independent of the game, second reading | 0.612 | 0.614 | +0.001 (0.005) |
| SkillCorner, live play only (reported, not in the rule) | 0.810 | 0.785 | -0.025 (0.004) |

In the SkillCorner whole-match score the frames with fewer than 14 detected players are
neutralised, the same for both columns, as stated before any result: there the camera is away from
the pitch (36.5 % of SkillCorner's frames have no detected player at all), so a labelled stoppage
is given as found and a stoppage predicted where the label says play counts as an abstention.

| stoppages on SkillCorner, frames with at least 14 detected players | E7 | submitted |
|---|---|---|
| recall of D | 0.21 | 0.88 |
| precision of D | 0.96 | 0.90 |
| live frames predicted D | 0.2 % | 2.4 % |
| predicted stoppages of 2 s or more that are real (they overlap a labelled stoppage or end at a restart event) | 0.989 | 0.985 |
| restart events with a predicted stoppage in the 5 s before | 0.28 | 0.92 |

How it was selected, in order, all on 13 November and with external data only:
1. A first rule, fixed before any classifier was trained, compared four candidates (trained on
   SkillCorner only, on DFL only, on both, and a recalibration of E7's own P(alive)) and asked for
   a whole-match score on SkillCorner better by more than 2 paired SE, a SkillCorner live score
   not worse by more than 1 paired SE, and DFL not worse by more than 1 paired SE in either
   reading. No candidate passed (experiments/alive_select.md): the one trained on both sources
   gained 0.088 on SkillCorner and 0.037 / 0.033 on DFL and lost 0.023 on SkillCorner live.
2. Two further checks of that candidate, fixed before measuring them
   (experiments/alive_checks.md): its predicted stoppages on SkillCorner are real (0.98), but with
   ball gaps independent of the game on 44 % of live play it predicted more false stoppages than
   E7 (11.5 % of the live frames against 7.9 %) and lost 0.012 with the second reading (2.2 paired
   SE). Not adopted.
3. The rule was then replaced, before the submitted classifier was trained. The live-score
   condition was dropped because that score cannot reward a detected stoppage: the frames
   SkillCorner labels as stopped are removed before scoring, so a correct stoppage earns nothing
   and every disagreement with an approximate label costs. Every candidate that found more
   stoppages lost on it, and so does E7 itself with other decoder costs
   (experiments/alive_posthoc.md). In its place the rule asks directly that the predicted
   stoppages are real, and adds the regime in which the previous candidate had failed. One
   candidate (the previous one plus the DFL copies with gaps independent of the game) and one
   training per fold, adopted only if: (1) whole-match spellF1 on SkillCorner better than E7's by
   more than 2 paired SE; (2) at least 0.85 of its predicted stoppages on SkillCorner real;
   (3) not worse than E7 by more than 1 paired SE, in either reading, on DFL with gaps independent
   of the game; (4) the same with the simulator's gaps. It passes the four (tables above).

What this selection is and is not. The second rule was written knowing the results of the first
screen: the previous candidate already met conditions 1, 2 and 4, so what was open was condition 3
and whether the new training copies would keep the others. The whole-match score on SkillCorner
rests on an approximate label made by another model. The reserved test had been queried once, with
E7, and was not queried again, so the classifier has development estimates only. It was trained,
selected and adopted with external data only.

## 7. Traceability

**What each match was used for.** Development folds: one DFL match and three or four SkillCorner
matches per fold (scripts/b1_dev.py `folds`); in each fold the last training DFL match and two
training SkillCorner matches are the early-stopping set.

| matches | simulator calibration | simulator choice | training | decoder choices (cross-validation) | reserved test |
|---|---|---|---|---|---|
| DFL J03WMX, J03WN1 | clean references (ball visibility per gap multiplier) | - | yes | yes | - |
| DFL J03WOH, J03WOY, J03WPY | - | - | yes | yes | - |
| DFL J03WQQ, J03WR9 | - | - | - | - | yes |
| SkillCorner 1886347, 1925299, 1927964 | target statistics | excluded | yes (team head; stoppage classifier) | yes | - |
| the other 14 SkillCorner development matches | - | yes (three levels) | yes (team head; stoppage classifier) | yes | - |
| SkillCorner 1874553, 1899585, 2006363 | - | - | - | - | yes |
| DFL event data (the same 7 matches) | - | - | - | second reading of the labels | second reading |
| BAR-ATH tracking | - | - | - | - | - (prediction only) |
| BAR-ATH Smart Tagging | - | - | - | - | - (label-free check that the second half is aligned like the first; aggregates only, nothing chosen) |

**Label-free reads of BAR-ATH other than prediction.** Automatic checks, aggregates only; nothing
was trained, tuned or chosen with them. Their scripts and reports are in the repository:
- experiments/domain_stats.md and dataset_audit.md (2 November): label-free statistics of the
  tracking of each source, BAR-ATH included (ball missing, gaps, tracked players, speeds, track
  fragments), to make the sources comparable.
- experiments/smart_tagging_halves.md (12 November): agreement of an output file with Metrica's
  automatic tags, as a check of alignment and of the home / away convention.
- scripts/dead_ball_check.py: share of predicted stoppages, set pieces of the Smart Tagging with a
  predicted stoppage nearby, and visible-ball episodes outside the lines without one. The last one
  is the origin of the ball-out rule (section 4).
- scripts/sanity_checks.py: attacking direction per period, and the labels predicted in the first
  and last 20 s of each graded half.
- 13 November: the output file of the submission was compared with the output file of E7 as a
  check of the integration of the stoppage classifier (share of each label, number of stoppages,
  frames with the same label; no label of the match is involved). The decision to adopt had
  already been taken by the rule of section 6.
- the comments of src/poss/tokens.py that give properties of BAR-ATH's tracking (tracks that
  fragment, a ball often missing) as the reason for tokens without identity and for the reference
  point come from the same inspection of 2 November.

**Two physical constants of the features**, derived from external data
(scripts/derive_constants.py, experiments/derive_constants.md):
- the impulse window, 0.2 and 0.4 s: one and two widths of the ball smoothing of broadcast
  tracking, 5 frames at 25 fps in the calibration on SkillCorner;
- the reach speed V_REACH = 7.0 m/s, the 99th percentile of player speed at 10 Hz: 6.76 on DFL
  and 6.51 on both external sources pooled, 7 when rounded to the whole m/s.

Both are physical scales, not tuned on any score (src/poss/physics.py).

**Discarded, and why.**

| decision | why |
|---|---|
| ball-position head (predicting where the ball is inside gaps) | tested in development, not finished in time; future work |
| rules on which team restarts | measured on development: the team accuracy in the 3 s after a restart (0.921) was already no worse than in open play (0.915), no restart type clearly worse except corners with few cases; no rule written |
| Smart Tagging as an input | no external match has it, so how much to trust it cannot be trained or tuned on external data; future work |
| the other stoppage classifiers (13 November; external data only; section 6) | trained on SkillCorner only: on DFL it predicts a stoppage on 27 % of the live frames (-0.10 and -0.12 against E7); trained on DFL only: it finds almost no stoppage on SkillCorner where at least 14 players are detected (recall of D 0.08); a recalibration of E7's own P(alive): a third of the gain on SkillCorner and -0.026 on DFL with the second reading; trained on both without the copies with gaps independent of the game: rejected by the first rule and by the check with many ball gaps |

## 8. Limitations and future work

- **Possession granularity is the main source of uncertainty.** The two labelled sources define
  possession differently, and the challenge's ground truth comes from a third convention. The
  switch cost sets the granularity; it was taken where the external sources are near their best
  at once, and how close that is to the target convention cannot be measured without its labels.
- **Short possessions** remain the weak point: they need evidence of the touch and of the
  change of owner at frame level, which the ball gaps of automatic tracking often hide. On the
  held-out development sets the submission keeps 52 % of the possessions shorter than 3 s on DFL
  (flag; 71 % with the second reading) and 59 % on SkillCorner, against 90 % and 91 % of the longer
  ones (experiments/short_spell_check.md). The decoder is not the main cause: 40 % of the short
  possessions of DFL and 23 % of SkillCorner's have no evidence in the probabilities (P(home |
  alive) is on the other team's side on most of their rows), and of those that do have it the
  decoder loses 13 % on DFL and 24 % on SkillCorner. A team-switch cost that adapts to the change
  head of E5 was to be tried only if that last share reached 25 % in some provider, a criterion
  fixed before measuring; it did not, so it was not tried.
- **SkillCorner's labels are another model's output**; its dead-ball label (the gaps between its
  phases of play) is only approximate. E7 does not train on it; the stoppage classifier of the
  submission does, and inherits its convention where it differs from DFL's.
- **Stoppages on real tracking are validated against one approximate label.** E7's own P(alive)
  did not transfer to real tracking (section 3). The stoppage classifier of the submission
  corrects that on SkillCorner (section 6), but the evidence is one provider (broadcast tracking of
  one league), a dead-ball label made by another model, and development folds only, with no
  reserved test. Where fewer than 14 players are tracked the classifier is supported by simulated
  DFL only (its SkillCorner rows are excluded there), and those frames are neutralised in its
  SkillCorner score. It is a per-row classifier with the context windows of B1, not a sequence
  model: a stoppage head trained inside the sequence model on the same data is future work.
- **Neither source alone gives a stoppage classifier that transfers to the other**
  (experiments/alive_select.md, the candidates trained on one source). Trained on SkillCorner only
  and applied to DFL with the dead ball hidden, it finds every stoppage (recall of D 1.00) but
  also predicts one on 27 % of the live frames (precision 0.76; AUC 0.96). Trained on DFL only (the
  dead ball hidden, without the copies with gaps independent of the game) and applied to
  SkillCorner where at least 14 players are detected, it finds almost none (recall of D 0.08,
  precision 0.97; AUC 0.77). Only the classifier trained on both works on both, so its level on a
  third provider, the tracking of the provided match, is not established by these data.
- **Simulator.** It is calibrated on broadcast tracking, not on the tactical camera of the provided
  match, which is why a range of intensities is trained instead of one.
- **Final models** are trained for a fixed number of steps without a validation set.
- **Future work, not in the submission:**
  - a ball-position head (predicting where the ball is inside gaps as an auxiliary task), tested in
    development but not finished in time;
  - Smart Tagging as an input (team in possession and set pieces logged by an operator would
    constrain the decoder directly); not used because no external match has it to train or tune
    how much to trust it;
  - broadcast tracking and video, to see the ball when the tactical camera loses it;
  - for short possessions, where the evidence and not the decoder is missing: a head that predicts
    duels and their winner, a learnt combination of the members instead of their mean (stacking),
    and a semi-Markov decoder with a mixture of durations.

## 9. Reproduction

    pip install -r requirements-solution.txt
    python scripts/fetch_dfl.py
    git clone https://github.com/SkillCorner/opendata.git data/external/skillcorner_raw/repo
    git -C data/external/skillcorner_raw/repo checkout 4340d274572876239c154c90bc507a9b3250a656
    python scripts/fetch_skillcorner.py --lfs && python scripts/fetch_skillcorner.py
    python scripts/make_submission_b2ens.py --team <team> --config configs/b2_ens_dr_alive.json   # submitted
    python scripts/make_submission_b2ens.py --team <team> --config configs/b2_ens_dr.json         # E7 as frozen on 12 November
    python scripts/make_submission.py --team <team> --config configs/b1_ext.json                 # B1 (reserve)

**Submitted file.** Inference only, from the saved weights of E7 in models/dr/ and the saved
stoppage classifier in models/alive/ (configuration in configs/b2_ens_dr_alive.json). CPU, a fixed
number of threads, `torch.use_deterministic_algorithms(True)`. The probabilities are rounded to
1e-4 before the decoder, so the decoded labels do not depend on floating-point summation order.
Three runs, one of them with a different number of threads, give the same file, SHA-256
236b8f603eccf74dcf08d3be22612cac997af8c8a66a066bc84dec19d845f7ac.

With configs/b2_ens_dr.json the same script gives E7 as frozen on 12 November (its own P(alive)),
SHA-256 df9bc4c947cc7999145c837434220ad7d6d1acab00c153f23c31362f5ecc5a93.

**Retraining the models.** scripts/b2_final.py retrains the weights of E7 (`--steps` with the
number of steps of each member, given in the configuration; training is not claimed to be
byte-reproducible). The stoppage classifier is retrained by

    POSS_SIM=ext_dr python scripts/alive_dev.py rows
    POSS_SIM=ext_dr python scripts/alive_dev.py rows_g
    POSS_SIM=ext_dr python scripts/alive_dev.py final g models/alive/alive_g.txt

(LightGBM with `deterministic=True` and a fixed thread count; a second training gave a byte-identical model file, SHA-256
b79f7b212f64f683364d2fc9a96be5061c87ebe6300c16566381dda76b5a29ac).

**Retraining from scratch.** The cached SkillCorner rows of development end one native frame before
the last (as the B2 tokens do); scripts/b1_dev.py rebuilds them that way (dataset.rows with
n_rows), and rebuilt from empty caches they are bit-identical to the cached ones. B1's submission
script computes its rows itself and does not depend on this.

**B1 (reserve).** Calibrates the simulator on SkillCorner, degrades the DFL training matches, trains
and predicts: CPU only, about an hour on 6 cores and under 4 GB of memory. LightGBM runs with
`deterministic=True` and a fixed thread count. Three runs from empty caches give the same file,
SHA-256 57469adf70d608ded7a07f4b823209bed1f91074fa4c5db60679493626205553.

**Cross-machine determinism.** Regenerating on a second machine (Windows 11, Intel AVX-512, against
Linux on AMD AVX2) first differed in 5 frames; the cause was a discrete choice, not floating-point
noise: `torch.topk` breaks exact ties differently on the two instruction sets when E1 picks its
nearest players. A stable sort (ties broken by slot index) made both machines produce the same
file; it is now the model's behaviour, and the ordering of players in the tokens uses a stable sort
too.

The submitted file was regenerated on both machines (Linux on AMD AVX2, Windows 11 on Intel
AVX-512) with the same result, SHA-256 236b8f60...; so was E7 alone (df9bc4c9...). One defect was
found and fixed on the way: on Windows, git's end-of-line conversion rewrote the saved stoppage
classifier, a text file, and LightGBM refused to load it. `.gitattributes` now excludes models/
from any conversion, and the script reads the model as text, so it loads with either line ending.

**Development scripts.** scripts/b1_dev.py and scripts/b2_dev.py (cross-validation),
scripts/sim_ext_calibrate.py, sim_ext_dr.py and sim_ext_select2.py (simulator), xfold.py (costs
chosen excluding the evaluated fold), f4_infer.py and f4_rule.py (inference variants), decoder_select.py (decoder), test_once.py
(reserved test), max_per_team.py, smart_tagging_halves.py (alignment check), gap_structure_check.py,
alive_dev.py, alive_select.py, alive_checks.py and alive_select2.py (stoppage classifier). Results are in experiments/ (aggregates only).
