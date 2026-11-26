# Delhivery Last-Mile Delay Intelligence Pipeline

**Reference implementation for learning and verification, newly generated on 24 November 2026.** This is not evidence of work uploaded before the assessment deadline. The assessment requires the student's own code and README. The provided generator was reconstructed from the PDF because some lines were clipped; its key numerical checks match the published reference, but byte identity with the original generator cannot be established. Replace it with the original verbatim script from the assessment platform when reviewing your own implementation.

This offline project uses synthetic data, not actual Delhivery records. It estimates delivery duration, predicts delayed parcels at dispatch, segments hubs, and assigns operational actions.

## How to run

Use Python 3.12 or another version compatible with the pinned dependencies. From the repository root, create a virtual environment if desired. Execute the commands in order:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

The generator creates 24 hub rows, 1,224 historical rows and 48 new rows. `src/shipment_prep.py` centralizes the cleaning and feature logic. The modelling table is consumed by the regression, classification and segmentation scripts; the classifier configuration and segmentation outputs are consumed by the final scoring workflow. All 22 results CSVs are in `outputs/`. `python package_model.py` regenerates `models/delay_model.pkl`. The committed pickle should be loaded only with the pinned scikit-learn version. If the installed version differs, the grader re-runs the packaging script instead of loading the committed pickle. Only load trusted pickle files.

## Data findings

The checks in `outputs/part_a_checks.csv` show 1,224 raw rows, 24 duplicates, 1,200 unique shipments, 278 delayed parcels and a delay rate of 0.2317 (23.17%). There are no missing cells in the model features or targets after cleaning. Five weight readings above 100 kg were treated as missing, not valid weights. The filled distance median is 314.95 km and weight median is 1.83 kg.

`outputs/part_a_delay_by_group.csv` establishes the following:

- Express has the highest delay rate, 0.4897 (119 of 243). Standard has rate 0.2056 and Economy 0.0857. The tighter Express promise makes it especially sensitive to delay.
- Tier 1, 2 and 3 delay rates are 0.0400, 0.2150 and 0.4400 respectively. The administrative tier is a useful broad indicator but not a perfect operational segment.
- Delay rates rise from 0.0219 for 0–200 km to 0.1368 for 201–400 km, 0.4364 for 401–600 km and 0.8523 above 600 km. Long routes warrant closer monitoring.

In `outputs/part_a_correlations.csv`, the strongest absolute correlations with delivery hours, excluding delivery hours itself, are promised_hours (+0.8372), distance_km (+0.4013) and n_stops (+0.3247). These associations are not causal estimates. The filled declared value mean is INR 1,533.0542 and median INR 1,121.50. After natural-log transformation the mean and median are 7.0277 and 7.0224. The transformation compresses large values and reduces the mean–median gap.

## Model results (regression)

`outputs/part_b_checks.csv` records the 840/180/180 split. The scaler is fitted on training rows only. `outputs/part_b_metrics.csv` records a mean baseline validation R² of -0.0212, indicating a constant training mean does worse than using the validation mean as a reference. Linear regression wins on validation with RMSE 4.8912 hours and R² 0.9313. Its untouched test evaluation gives RMSE 4.5235 hours and R² 0.9493.

In `outputs/part_b_tree_depth.csv`, depth 1 has training RMSE 11.6334 and validation RMSE 12.1324: both are high, indicating bias. Depth 6 has the lowest validation RMSE, 6.2796. An unlimited tree has zero training RMSE but validation RMSE 7.8368, showing overfitting; depths 8 and 12 also show growing gaps. In `outputs/part_b_gb_learning_rate.csv`, learning rate 0.1 performs best (R² 0.9306), while 0.01 learns too slowly for the fixed tree count (0.7127) and 1.0 takes overly aggressive steps (0.8440).

`outputs/part_b_cv.csv` shows gradient boosting has higher cross-validated mean R² (0.9353 versus linear regression 0.9261). The prescribed final test model remains the validation winner, linear regression. For an initial deployment, linear regression is a defensible simple, fast and interpretable choice given its slight validation RMSE advantage; the stronger CV result for boosting merits additional time-based evaluation before a production decision. Neither model is selected using test performance.

## Model results (classification)

The stratified split contains 195, 42 and 41 delayed shipments in training, validation and test respectively. From `outputs/part_c_models.csv`, the blind always-on-time baseline reaches accuracy 0.7667 while detecting none of the 42 validation delays. Accuracy alone is misleading. Logistic regression achieves validation F1 0.8046; the selected decision tree at depth 6 reaches 0.7470; the default full-feature forest reaches 0.7848.

In `outputs/part_c_feature_scores.csv`, the bookkeeping columns contribute 0.2245 combined forest importance (sum of rounded output values), yet all five rank outside the top seven by F-score. Forest importance may allocate signal to irrelevant variables, whereas the F-score distinguishes the seven strong features: distance, hub tier, capacity, stops, promised hours, rain and sale event. `outputs/part_c_selectk.csv` gives F1 of 0.6988, 0.8333, 0.7857, 0.8049 and 0.7848 at k=5, 7, 10, 12 and 17 respectively. Seven features are preferred.

`outputs/part_c_tuning.csv` shows grid search and randomized search have best CV accuracies 0.8952 and 0.8905 and validation F1 of 0.7750 and 0.7848. The default seven-feature forest wins with 0.8333 validation F1. Search is not guaranteed to improve a model; a small validation set and different optimization objectives can change rankings. The final configuration is the 100-tree default forest with seven selected features.

`outputs/part_c_thresholds.csv` selects 0.5 as the highest threshold with validation recall at least 0.80. At this threshold the validation precision is 0.7955 and recall 0.8333. Increasing to 0.6 improves precision to 0.8889 but reduces recall to 0.7619, violating the operations rule. Note that model.predict and thresholding predict_proba with >= can differ on exact 0.5 ties.

`outputs/part_c_final_test.csv` records precision 0.6889, recall 0.7561 and F1 0.7209: 31 true positives, 14 false positives, 10 false negatives and 125 true negatives. Test recall is lower than validation recall, so the validation target should not be represented as a production guarantee. With only 41 delayed test parcels, each missed parcel materially changes recall.

## Operations recommendation

The three segments in `outputs/segment_profile.csv` are Standard (7 hubs, delay rate 0.0257), Watch (8 hubs, 0.1550), and Escalate (9 hubs, 0.4600). Prioritize high-risk hubs using observed operating performance rather than administrative tier alone.

Escalate hubs and historical delay rates, from `outputs/hub_segments.csv` joined with `outputs/hub_profile.csv`:

| Hub | City | Delay rate |
|---|---|---:|
| HUB13 | Nagpur | 0.4600 |
| HUB15 | Kochi | 0.3600 |
| HUB17 | Patna | 0.4600 |
| HUB18 | Ranchi | 0.3800 |
| HUB19 | Guwahati | 0.6200 |
| HUB20 | Raipur | 0.4800 |
| HUB21 | Bhubaneswar | 0.4000 |
| HUB23 | Siliguri | 0.4400 |
| HUB24 | Agartala | 0.5400 |

`outputs/tier_vs_segment_mismatch.csv` identifies Kolkata (HUB07, Tier 1, Watch), Nagpur (HUB13, Tier 2, Escalate), Kochi (HUB15, Tier 2, Escalate), and Dehradun (HUB22, Tier 3, Watch). Kolkata needs more oversight than its formal tier suggests. Nagpur and Kochi merit escalated interventions. Dehradun can follow the Watch playbook rather than automatic escalation based on tier.

Keep K=3 because operations needs three viable playbooks. `outputs/part_d_separation.csv` gives across-to-within ratios 2.8940, 5.3929, 6.3048 and 6.8867 for K=2,3,4,5. Although K=4 and K=5 improve separation, their cluster sizes 8-7-5-4 and 8-7-4-3-2 each include groups smaller than six hubs; K=3 has 9-8-7 and balances separation and coverage.

At the selected threshold 0.5, roughly 6.9 of every 10 flagged parcels are truly delayed and about 7.6 of every 10 delayed parcels are caught (test precision and recall). `outputs/new_shipment_actions.csv` contains 48 new parcels: 8 flagged, of which 6 require escalation to a hub manager and 2 require proactive customer calls; 40 require no action. This action list is a model recommendation, not a known future outcome. `outputs/part_d_reload_check.csv` records a maximum absolute probability difference of zero after model reload for 180 test rows.

## Production plan

- Accuracy: use the held-out regression RMSE and classification precision, recall and F1 tables as initial baselines.
- Reliability: check the reload probability difference and validate unique IDs, known hub IDs and complete model inputs.
- Reproducibility: retain pinned package versions, fixed random seeds, data generator and ordered scripts.
- Performance: score shipments in batches with vectorized preprocessing and a compact seven-feature forest; measure latency before setting an SLA.
- Maintainability: keep preprocessing in `src/shipment_prep.py` and separate model development, packaging and inference.
- Consistency: scoring uses the saved medians, feature order and threshold, with the same cleaning functions as training.
- Monitorability: log prediction probabilities, flags and actual outcomes when deliveries finish, grouped by service, hub and event type.

Proposed operational triggers (design targets, not measured results): investigate weekly actual delay rate above 30%; investigate weekly flagged share above 35% or below 10%; review threshold and data quality if observed recall drops below 70%; investigate if any input missingness exceeds 5%. Compare comparable ordinary weeks and compare the 15–28 September sale window with other sale events, not ordinary weeks. Drift could appear as changes in distance, stops, service mix, weather or missing fields, or changed delay outcomes for similar inputs. Review monthly and retrain quarterly or sooner if persistent drift is confirmed. Use chronological holdout data and only approve a replacement after quality checks.

## Limitations

The dataset is synthetic and deliberately simplified; real logistics outcomes have additional causes. A random split does not establish generalization to a later month or a new hub. The small test set has only 41 delayed parcels. The threshold is tuned on one validation split, and the forest probabilities have not been calibrated on independent data. The three hub clusters reflect one historical period, not immutable hub characteristics. The reconstructed data generator cannot be certified byte-identical to the original PDF-clipped source. The reference implementation is for study and debugging; it does not establish eligibility for the original assessment or represent the student's independent work.
