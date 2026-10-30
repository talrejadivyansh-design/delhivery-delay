# Delhivery Delay Intelligence
## Data findings
1. **Service type.** Express Parcels are delayed most often at a delay rate of 0.4897(48.97%).
2. **Hub Tier.** Delay rate of Tier 1 is 0.04 and it changed to 0.215 in Tier 2 while in Tier 3 it became 0.44.
3. **Distance Bands.**  Delay Rate goes from 0.0219 for (0-200km) to 0.1368 for (201-400km) to 0.4364 for (401-600km) to 0.8523 for (601+km)
4. **Correlations with Delivery Hours.** The three strongest are promised_hours with a correlation of 0.8372 and distance_km with 0.4013 correlation with delivery_hours and n_stops with a correlation of 0.3247.
5. **Log transform.** declared_value_inr was log-transformed because it is right-skewed. A few big values pull the mean up: the mean is 1533.0542 and the median is 1121.5. After the log, the mean is 7.0277 and the median is 7.0224, which are almost equal.
## Model Results(regression)
1. **Baseline.** The baseline r2 is -0.0212 which explains no variation every model has to beat this number.
2. **Winner On Validation.** Linear Regression won on validation and it's rmse is 4.8912 which is almost 4.9 hours of error.
3. **High bias and high variance.** Depths 1, 2 and 3 are high bias. The training error is already high (11.6334, 9.3687, 7.5295) and the gap to validation is small (0.499, 1.04, 1.17). The tree is too simple to fit even the data it has seen. Depth 12 and unlimited are high variance. They are so complex that they memorise the training rows. The training error is small (1.3613 and 0.0000) but the validation error is big (7.4173 and 7.789).
4. **Learning rate.** A learning rate of 0.1 was best (validation R² 0.9306, RMSE 4.9165). At 0.01 each of the 100 trees corrects too little, so the model is still underfitting when it runs out of trees (R² 0.7127). At 1.0 each tree corrects too much and overshoots the best fit (R² 0.8440).
5. **Which model I would ship.** The cross-validation ranking disagrees with the validation ranking. Cross-validated mean R² is 0.9352 for gradient boosting against 0.9261 for linear regression, while linear regression has the lower validation RMSE. I would ship gradient boosting: one split of 180 rows can favour a model by luck, while 5-fold cross-validation over 1,020 rows averages that luck out.
## Model Results(classification)
1. **Blind baseline.** Its accuracy is 0.7667, which looks fine, but it catches none of the delayed parcels. TN is 138 and FN is 42. So it is useless.
2. **Noise Columns.** Combined Importance is 0.2245 and all are below the top 7 one is on 8 other on 12 and 14,16 and 17 and each f-score below 10 highest is (4.5847).
3. **K-Table.** Validation F1 peaks at k = 7 (0.8333). With fewer columns (k = 5: 0.6988) real signal such as rain_flag and sale_event_flag is dropped, and with more columns (k = 10: 0.7857, k = 12: 0.8049, k = 17: 0.7848) noise columns are added. So k = 7 was useful so we used that after.
4. Tuning didn't won on untuned random forest as grid-search scored 0.775 and random-search scored 0.7848 against 0.8333 for the untuned forest on the same 7 columns. The gain over the 17-column forest (0.7848 to 0.8333) came from dropping the ten low-signal columns with SelectKBest, not from tuning the forest's settings. So on a 180-row validation set these scores are too close to say that tuning helped or not.
5. **Alert threshold.** The alert threshold is 0.5, the highest threshold whose validation recall is at least 0.80. At 0.5 validation recall is 0.8333 and precision is 0.7955. Raising it to 0.6 gives a precision of 0.8889, but recall falls to 0.7619, below the 0.80 floor. Lowering it to 0.4 gives recall 0.8810, but precision falls to 0.7400 and more false alarms follow.
6. **Test result.** On the test rows at threshold 0.5 the model reached precision 0.6889, recall 0.7561 and F1 0.7209. Test recall (31 of 41) is lower than validation recall (35 of 42, 0.8333). With only 41 delayed parcels in the test set, that gap is within what luck can produce, so I treat the test recall as a rough estimate and not as evidence that the model got worse.
## How To Run
Run these commands one after another from the project folder:
```
git clone https://github.com/talrejadivyansh-design/delhivery-delay.git
cd delhivery-delay
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```
The order matters because each script uses files made by the one before it. generate_data.py makes the raw data in the data folder. part_a_clean_eda.py cleans it and saves data/modelling_table.csv. The part B, C and D scripts save their results in the outputs folder. package_model.py saves the model and its helper files in the models folder. predict_delay.py then scores the 48 new parcels and saves outputs/new_shipment_actions.csv. If you run predict_delay.py twice, you get the same file both times.
requirements.txt has pandas 3.0.5, numpy 2.4.4, scikit-learn 1.9.1 and joblib 1.6.0. The file models/delay_model.pkl is made again every time you run python package_model.py. The saved pickle only loads safely on the scikit-learn version in requirements.txt, so if your version is different, run package_model.py again and don't use the saved pickle.
## Operations Recommendation
1. **Escalate hubs.** 9 hubs are in the Escalate segment, and their average delay rate is 0.46. They are HUB13 Nagpur (0.46), HUB15 Kochi (0.36), HUB17 Patna (0.46), HUB18 Ranchi (0.38), HUB19 Guwahati (0.62), HUB20 Raipur (0.48), HUB21 Bhubaneswar (0.40), HUB23 Siliguri (0.44) and HUB24 Agartala (0.54). These hubs have the longest routes (about 470 km) and the most stops per parcel (about 84), so hub managers should look at them first. Standard hubs have a delay rate of 0.0257 and Watch hubs have 0.1550.
2. **Tier labels.** 4 hubs do not match their tier. HUB07 Kolkata is Tier 1 but behaves like a Watch hub. HUB13 Nagpur and HUB15 Kochi are Tier 2 but behave like Escalate hubs. HUB22 Dehradun is Tier 3 but behaves like a Watch hub. So the tier label should not be used to judge how a hub really performs. The segment is a better guide.
3. **Alert threshold.** The alert threshold is 0.5, which means a parcel is flagged when its chance of delay is 0.5 or more. On the test parcels, precision is 0.6889, so about 7 out of 10 flagged parcels are really delayed. Recall is 0.7561, so we catch about 7 to 8 out of every 10 delayed parcels. The test set has only 41 delayed parcels, so these numbers are only a rough idea.
4. **New parcels.** Out of the 48 new parcels, 8 were flagged. 6 of them are in Escalate hubs (HUB24, HUB18, HUB17, HUB20 and two at HUB21) and need to be escalated to the hub manager. The other 2 (at HUB22 and HUB09) just need a customer call. The other 40 parcels need no action.
5. **Why K = 3.** In part_d_separation.csv the ratio keeps going up as K increases (2.894, 5.3928, 6.3047, 6.8866 for K = 2 to 5). But K = 4 has a cluster of only 4 hubs, and K = 5 has clusters of 4, 3 and 2 hubs. Those groups are too small to have their own playbook, and operations runs exactly three playbooks. So I kept K = 3, which gives groups of 9, 8 and 7 hubs.
## Production plan
Which production-readiness points are covered, and where:
- **Accuracy:** outputs/part_c_final_test.csv holds the test precision, recall and F1, and outputs/part_b_metrics.csv holds the regression errors.
- **Reliability:** clean_shipments fills missing values with stored medians (models/impute_values.csv), so a new parcel with a blank value does not crash the scoring. outputs/part_d_reload_check.csv shows that the saved model gives the same probabilities as the one in memory (difference 0.0).
- **Reproducibility:** requirements.txt pins the exact library versions, every random_state is 42, and two runs of predict_delay.py give the same file.
- **Maintainability:** the cleaning and feature code lives in one file, src/shipment_prep.py, and each part has its own script.
- **Consistency:** training and scoring both use the same clean_shipments and build_features functions, and the column order is saved in models/feature_order.csv.
- **Monitorability:** outputs/new_shipment_actions.csv keeps the probability, the flag and the action for every parcel, so each week's numbers can be counted and compared.

What to monitor, with a trigger for each:
1. **Weekly delayed share of delivered parcels.** In the training data it is 0.2317. Trigger: below 0.15 or above 0.31 for two weeks in a row.
2. **Weekly share of parcels flagged.** On the test rows the model flagged 45 of 180 (0.25), and on the 48 new parcels it flagged 8 (0.17). Trigger: below 0.15 or above 0.35 in a week.
3. **Recall on parcels whose outcome is now known, checked every month.** Test recall was 0.7561. Trigger: below 0.65.
4. **Share of missing values in the new parcels' distance, weight, value and payment mode.** In the new file, 3 of 48 distances were missing. Trigger: any column above 10% missing in a week.
5. **Retraining.** Retrain every month using the latest parcels with known outcomes, and also straight away if any trigger above fires two weeks in a row. After each retrain, run all the scripts again and compare the new test numbers with the old ones before replacing the model.
6. **Sale window.** The 15 to 28 September sale (sale_event_flag) will push delays up, so a normal week is the wrong thing to compare it with. Compare it with the same kind of event, which means the last sale window, or with other parcels that have sale_event_flag = 1. A jump in delays inside the window should not trigger a retrain by itself. After the window ends, add the sale parcels to the training data so the model learns from them.
7. **Data drift.** Here it would look like: a different mix of service types (more Express), a shift in distance or number of stops, new hubs or changed hub capacity, new spellings or more missing values in the raw file, or the flagged share and delayed share slowly moving away from the numbers above.
## Limitations
1. **The data is synthetic.** It was made by a script with rules, so the patterns are cleaner than real parcel data, and the results may not hold on real shipments.
2. **The test set has only 41 delayed parcels.** One parcel changes recall by about 0.024, so the test precision and recall are only rough.
3. **The threshold was picked on one validation split.** It has 180 rows with 42 delayed parcels. A different split could give a different threshold, and the move from 0.5 to 0.6 is only a few parcels.
4. **There is no real cost for a missed parcel versus an extra call.** The rule "recall at least 0.80" is a simple stand-in, not a number from a cost calculation.
5. **Hub segments come from only 24 hubs and 50 parcels per hub.** One odd hub can change the clusters.
6. **The data is split at random, not by date.** The model was never tested on a future time period.

