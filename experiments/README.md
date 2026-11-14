# Reports: which pipeline each one describes

Aggregates only.

## 1. The pipeline

| report | what it is |
|---|---|
| sim_ext_calibration.md | calibration statistics of the external simulator |
| sim_ext_select.md | first choice of intensity (two folds); superseded by sim_ext_select2.md |
| sim_ext_dr.md | table gap-onset multiplier -> visible ball of the domain-randomisation arm |
| sim_ext_select2.md | choice of the simulator arm (technical document, section 5) |
| b1_ext_costs.md | B1 costs with the arm of sim_ext_select.md; superseded by b1_ext_dr_costs.md |
| b1_ext_dr_costs.md | B1 (reserve) costs with the chosen arm |
| max_per_team.md | player slots per team from the external data |
| f4d_rule.md | inference variants of E7 (team swap adopted) |
| decoder_select.md | decoder of the submission |
| b2_e7_final.md | E7 against each member on the final out-of-fold predictions |
| test_once.md | the reserved test, queried once |
| derive_constants.md | two feature constants derived from external data |
| smart_tagging_halves.md | label-free alignment check of the reserve file (B1) on BAR-ATH |
| gap_structure_check.md | stoppage detection of the frozen E7 by ball-gap regime and on SkillCorner (after the freeze; nothing chosen) |
| alive_select.md | replacing P(alive) only: four candidates under a first rule fixed beforehand; none adopted |
| alive_posthoc.md | exploratory, after alive_select.md: the loss on SkillCorner live does not come from the decoder costs |
| alive_checks.md | two further checks of the best of those candidates, fixed before measuring; one fails, not adopted |
| alive_select2.md | the stoppage classifier of the submission (candidate g) under the second rule, fixed before it was trained; adopted (technical document, section 6) |
| short_spell_check.md | short possessions of the submission: evidence in the probabilities against losses in the decoder; criterion for an adaptive switch cost not met, nothing tried |

## 2. Independent of any model

| report | what it is |
|---|---|
| domain_stats.md, dataset_audit.md | label-free statistics of the tracking of each source, BAR-ATH included |
| test_evaluate.md | tests of src/poss/evaluate.py against the organisers' scorer |
