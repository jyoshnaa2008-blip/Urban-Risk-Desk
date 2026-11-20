# Project Handoff

### Integrated Urban Risk Analytics and Resource Prioritization System

Written on Sunday 15 November 2026 so a new AI assistant with no memory of this project can continue it. Everything below comes from our earlier conversation, the student's files and checks run on them. Anything not verified is marked as such.

## 0. Read this first (instructions for the new AI)

- **The user is a real student** on a team of 4 with a deadline on **21 November 2026**. Marks matter a lot to them. Early in the old chat they asked the assistant to role-play as a student being marked by them. That role-play is over: speak to them directly now.
- **Style:** short, simple English. They sometimes ask for 'explain like I am 5'. They write casually with typos. Ask at most one question at a time. Do the work instead of only describing it.
- **Honesty rules:** never invent numbers; say what was not verified; report weak results as weak. One earlier number was retracted (see section 6). The student wants good marks, and an evaluator will punish inflated claims more than modest honest ones.
- **Ignore these obsolete ideas from the early chat** (they were brainstorming for a different plan): road accidents with UK STATS19 data, fusing three datasets on one H3 grid, ONNX in-browser inference, conformal prediction, difference-in-differences, and the 'India atlas' prototype. The current plan is NYC 311 as the main problem and UCI air quality as the second.

## 1. The assignment

Project 17: **Integrated Urban Risk Analytics and Resource Prioritization System** (shown on a second page as '-II'). Category: Web Development. Tech stack: JavaScript, ML. Maximum groups 5, and all 5 are enrolled.

**Problem statement (verbatim):** Analyze an urban problem using a public dataset such as road accidents, air pollution, or public service complaints. Perform two complementary tasks: predict a documented outcome using supervised ML and discover recurring risk profiles using clustering. Compare algorithms, explain factors associated with higher risk, validate on unseen data, and convert findings into a transparent resource-prioritization strategy.

**Reference links given:** UCI Air Quality (archive.ics.uci.edu/dataset/360/air+quality) and NYC 311 Service Requests (data.cityofnewyork.us, dataset id erm2-nwe9). The brief says 'an urban problem', so one dataset done well is enough. Using both is a bonus.

**What the evaluator will look for:** a sensible prediction target; at least three algorithms compared with proper metrics (not just accuracy); validation on later, unseen data; named and stable clusters; factor explanations (associations, not causes); a visible and adjustable prioritization rule; a working web app; honest limits.

## 2. Deadline and instructor notes

Raw notes the student typed from the instructor (interpretation in brackets):

- Host it on Kaggle. 21 Nov: submit the **URL of the hosted model**.
- We should know: who the stakeholders are; how we will use the model; how the output is useful; 'google api to check the conditions?' (unclear whether required).
- Principle 1: train on 'respective servers'? fine-tuning; convert it into FastAPI. [We read fine-tuning as hyperparameter tuning, since we use scikit-learn models, not pretrained ones.]
- Principle 2: create the UI with JavaScript, in Kaggle, using the fetch API; a control page with event handling.
- Principle 3: 'she will teach?' a CRUD application and how middleware talks to SQL and Python API calls to retrieve and store data.

**To confirm with the instructor:** (a) what exactly counts as the 'URL of the hosted model': a live API URL or a Kaggle notebook/model page; (b) whether the Google API is required; (c) what 'fine-tuning' means; (d) what the CRUD/SQL session will cover.

**Remaining time from 15 Nov:** six days.

## 3. Decisions already made

- **Main problem: NYC 311.** The student chose it ('this data set is fine'). Reasons: it has locations, agencies and complaint types, which give a real 'where to send crews first' strategy, and it fits a zoomable map.
- **Second dataset: UCI air quality**, already analysed. It is also the fallback if 311 breaks.
- The assistant had suggested air quality as the safer main problem because of the 7-day deadline. The student kept 311. Both pipelines share the same ideas (time-based split, model comparison, explanations, prioritization).
- **One problem in depth beats several shallowly.** The brief asks for one urban problem.

## 4. Data in detail

### 4a. NYC 311 export (main)

File **export.csv** (about 212 MB, 48 columns, some with blank names). 259,858 rows covering **all boroughs**, complaints created from **16 Oct 2026 18:42 to 8 Nov 2026 14:03**. This is much less than the six-month Brooklyn slice that was originally requested. The student chose to keep this file.

**Problems found and handled:**

- **5,244 junk rows** at the top: every cell contains the same number, 71044266. They look like an export glitch (cause unconfirmed). Remove them. This is the 'dataset changed to numbers' the student noticed.
- **Two date formats mixed:** '11/08/2026 02:03:00 PM' and '11-08-2026 14:03'. About 30 percent of rows use the second. A default parse silently drops them. Parse both.
- 170 complaints close before they were created (dropped); about 4,600 close in zero hours; about 5,000 rows are not marked closed but have a closed date; the Due Date column is filled in only 868 rows, so it cannot define lateness.
- A column was renamed: 'Problem (formerly Complaint Type)'. The brief's link title says 2010 to Present, but a web search showed the dataset is now titled 311 Service Requests from 2020 to Present, with 2010 to 2019 in a separate dataset.
- Result after cleaning: **254,439 rows**. Status counts (approx.): Closed 206,010; Open 28,835; In Progress 19,013; Assigned 650; Pending 86. Closing time among closed complaints: median 3.9 hours, 75th percentile 36.6 hours, 90th percentile 105 hours. Borough counts in the file: Brooklyn 82,281, Queens 58,116, Manhattan 56,327, Bronx 48,844, Staten Island 8,779.

**Cleaned file provided:** nyc311_clean.csv (41 MB, 254,439 rows). Columns: unique_key, created_date, closed_date, agency, complaint_type, descriptor, location_type, incident_zip, borough, status, channel, latitude, longitude, hours_to_close, slow_48h (ISO dates). 'open' means the agency has not marked the case finished; 'closed' means it was marked finished (not necessarily fixed).

### 4b. UCI Air Quality (second dataset)

File AirQualityUCI.csv (also .xlsx) in air_quality.zip. Semicolon separated, decimal commas, two empty trailing columns, **-200 means missing**. 9,357 hourly rows, 15 real columns, one Italian city, **10 Mar 2004 18:00 to 4 Apr 2005 14:00** (the UCI page says March 2004 to February 2005, so the real range should be stated in the report). Columns: Date, Time, CO(GT), PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC), NOx(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH. GT columns are reference-analyzer readings; PT08 columns are low-cost metal-oxide sensor responses. The data page restricts use to research purposes.

- NMHC(GT) is 90.2 percent missing, so drop it. CO, NOx and NO2 reference readings are about 17.5 to 18 percent missing. 366 rows (3.9 percent) have every sensor missing at once. Longest NO2(GT) gap: 173 hours.
- **Sensor drift:** monthly correlation of the NO2 sensor with the NO2 reference ranges from 0.45 to 0.84; CO sensor 0.84 to 0.95; benzene sensor about 0.98 (so benzene should not be treated as independent ground truth).
- Share of hours with NO2 above 150 micrograms: about 20 percent overall; 8 percent in training months (Mar to Oct 2004) but 41 percent in test months (Jan to Apr 2005). The EU hourly limit of 200 is exceeded in only about 5 percent of hours.

## 5. Air quality results (real, from our checks)

Target: NO2(GT) above 150. Features: sensor responses, T, RH, AH, hour, day of week, 1 to 3 hour sensor lags (no reference readings). Time split: train before 1 Nov 2004, test from 1 Jan 2005. Metric: PR-AUC.

| Task | Logistic | Random forest | Gradient boosting | Last reference reading |
|---|---|---|---|---|
| Current hour from sensors | 0.82 | 0.83 | 0.77 | not run |
| Next hour from sensors | 0.80 | 0.81 | 0.78 | 0.91 |

- A random split inflates scores (boosting 0.89 on the current hour against 0.77 on the time split), so time-based splitting is essential.
- **Negative result to report:** copying the last reference-analyzer reading beats every sensor-only model at next-hour forecasting. The sensors only add value where no analyzer exists.
- **Clustering** of 364 full days by hourly benzene profile: 2 clusters are very stable (silhouette 0.34, bootstrap agreement 0.94); 3 clusters are weaker (0.23 and 0.68). The 3 groups: 170 weekday-heavy days with the highest pollution (mean benzene 13.7, mostly autumn); 67 weekend-heavy days with the lowest (4.8); 127 mixed days (8.1). Evenings peak around 7 to 8 pm.
- Average NO2 by weekday (Mon to Sun): 110, 117, 120, 121, 123, 108, 92.

## 6. NYC 311: the machine learning task

**y (the answer to predict)** is built from two columns, because the raw file has no y column: **slow_48h = 1** if the complaint was closed more than 48 hours after it was created or has no closed date; otherwise 0. Only complaints created at least 48 hours before the end of the data (on or before 6 Nov 2026 14:03) have a known y: 235,808 rows, of which **34.7 percent are slow**.

**X (what the model may see)** must be known when the complaint is filed: complaint type, agency, borough, ZIP, channel, hour, day of week, and the recent slow rate of the same type and agency (a trailing 7-day window that ends 3 days earlier, needing at least 20 complaints). **Never put closed_date, hours_to_close, status or resolution text in X**: that leaks the answer.

**Key finding 1:** across all complaint types the task is nearly trivial, because the agency and type decide almost everything. Agency slow shares: NYPD 0%, DHS 3%, DOB 19%, DEP 44%, DSNY 48%, DOT 62%, HPD 75%, DPR 84%, DOHMH 88%. A lookup of each type's past slow rate scores PR-AUC 0.893 and the best model 0.911, so the model adds almost nothing.

**Key finding 2:** so we test within the complaint types where outcomes genuinely vary. These were chosen using the **training period only** (slow share between 0.15 and 0.85, at least 1,500 training rows). That leaves **3 types: HEAT/HOT WATER, NOISE, STREET CONDITION**.

**Split by date:** train before 25 Oct 2026; validation 25 to 27 Oct (used to choose the model and calibrate); test from 28 Oct (the final exam, used once).

| Model (test weeks, PR-AUC) | Score | Note |
|---|---|---|
| Baseline: type's slow rate from training | 0.399 | Random guessing would be about 0.34 (test slow share) |
| Baseline: type's slow rate over the last week | 0.339 |  |
| Boosting, basic features | 0.460 |  |
| **Boosting + recent slow rate (final)** | **0.507** | Chosen on validation weeks, not test. ROC-AUC about 0.62 against 0.55 for the baseline. Flagging the top 20 percent gives about 1.5 times better than random. |
| Boosting + recent rate + backlog features | 0.356 | Hurt. The file starts on 16 Oct, so early backlog counts are artificially low. |

**Retracted claim:** earlier in the chat the assistant reported 0.61 against a 0.52 baseline. That used 9 complaint types chosen with the whole file, including the test weeks, so it was optimistic. The honest numbers are the ones above.

**Drift (very important):** the slow share for heat and hot water by date was 0.21 (16 Oct), 0.64 (19 Oct), 0.73 (22 Oct), 0.70 (25 Oct), 0.24 (28 Oct), 0.33 (31 Oct), 0.09 (3 Nov), 0.12 (6 Nov). Across the 3 types the slow share fell from about 61 percent in training weeks to about 34 percent in test weeks. The cause is unknown. It coincides roughly with the date-format change in the export from 1 Nov, so a system or process change is a possible but unverified explanation. With only 3.5 weeks, no feature can learn a shift like this.

**Mitigations built:** the recent-slow-rate feature; daily recalibration (shift the log-odds so the average prediction matches outcomes already known, using out-of-sample rows from the last 7 days, at least 300 rows with both classes); show High, Medium and Low tiers instead of literal percentages; a drift alert. Even after recalibration the average prediction is 48 percent against 34 percent actual (Brier score 0.353 before, 0.238 after), so raw percentages should not be shown.

**Weak spot:** the priority table built on predicted risk did not beat ranking by raw complaint volume. The top 10 percent of (ZIP, type) groups hold 31.9 percent of slow complaints by model score, against 31.7 percent by volume.

## 7. Prioritization strategy and honest value story

**Priority score = predicted delay risk x complaint volume x impact weight.** The impact weights are human assumptions that the user can change (prototype defaults: heat and hot water 100, street condition 80, noise 50). There is a crews/budget control: the top N groups are highlighted and the screen shows the share of total priority they cover. Urgency is NOT learned: the data has no urgency label, and real emergencies go through 911.

**The 'why a model if departments exist?' question** (the student's mother asked it, and an evaluator may too). Routing is already done by 311, so the model does not do it. The delay happens after routing, inside queues. The model adds: (1) early triage inside one department, flagging complaints likely to stall at filing time, before they age; (2) one shared view across departments for splitting limited crews; (3) transparency. We cannot prove from this data that acting on the flags makes anything faster, and the gain is modest. Possible extension: forecast complaints per agency by hour to plan staffing.

**Stakeholders:** city operations officers (split crews and budget across agencies); department supervisors (triage their own queue); agency crews and inspectors; residents who file complaints; data and IT staff who recalibrate and monitor drift. For air quality: environment officers, transport authorities, public health teams and schools, residents, sensor operators.

**Google API question:** Google's Air Quality API exists (current conditions with hourly AQI and pollutants, forecast up to 96 hours), needs an API key and billing, and could feed an optional live-conditions panel. It cannot feed our models, which are trained on raw 2004 to 2005 sensor responses and 311 records. Keep any key on the server, never in JavaScript. Check with the instructor whether it is required.

**Clustering is not done for 311.** Plan: build profiles by ZIP, complaint type and time of day; compare K-Means and hierarchical (optionally Gaussian mixture); score with silhouette and Davies-Bouldin; check stability by re-clustering bootstrap samples; fit on the training period only; name each profile in plain words. Optional: test the cluster number as an extra feature for the predictor, using validation weeks, and keep it only if it helps. Never use y or the closed date for clustering.

## 8. Files produced

| File | What it is |
|---|---|
| pipeline_311.py | End-to-end script. Usage: python pipeline_311.py path/to/export.csv [output_dir]. Reads the RAW export (handles junk rows and both date formats), builds y and features, picks types from training weeks, compares variants, chooses the model on validation, recalibrates daily, writes the files below. |
| results.json | Variant scores, validation scores, chosen model, priority capture, list of chosen types. |
| priority_by_zip_type.csv | Test-period ranking of (ZIP, complaint type) groups with complaints, expected slow, actual slow, priority score. |
| model_311.joblib | Saved final model with feature list, chosen types and 48-hour threshold. |
| nyc311_clean.csv | Cleaned dataset with slow_48h (41 MB). |
| urban-risk-desk.html | Website sample (also published at claude.ai/artifact/6DcP2C6o1iVBNtyJoVY7Hr). An older, obsolete prototype sits at claude.ai/artifact/6JniBBgAob5TUoz9yzxmix and was replaced. |
| export.csv, air_quality.zip | The student's original raw datasets. |

The new AI cannot see these files. **Ask the student to re-attach them** (at least export.csv or nyc311_clean.csv, pipeline_311.py, results.json and the air-quality zip).

## 9. Website sample (Urban Risk Desk)

Dark control-room style (Unbounded and Hanken Grotesk fonts; magenta, cyan and yellow accents). Five pages: **Map** (stylised New York boroughs, click to zoom, filter heat/street/noise, hotspot detail); **Priority board** (visible formula, sliders for crews and impact weights, coverage percent); **Triage queue** (agency tabs HPD, DOT, DSNY, DEP; complaints ranked High/Medium/Low with a reason; Escalate button); **Models and drift** (real model bars, the drift line, what holds up and what does not); **Air quality** (real hourly NO2 chart, model comparison, day-type clusters, data checks).

**Real:** every number on the Models and Air quality pages, borough counts and agency slow shares. **Illustrative:** map hotspots, ZIP codes, queue items and risk scores. The 311 clustering is deliberately absent because it has not been run.

## 10. Architecture and hosting plan

```
Kaggle notebook (scikit-learn, train + tune)  ->  joblib model
   ->  FastAPI app:  POST /predict   GET /priority   GET /clusters   GET /metrics
                     CRUD /scenarios  (SQLite: saved weights, budgets, history)
   ->  JavaScript UI served by the same app, calling the API with fetch()
   ->  public URL through a tunnel (ngrok or cloudflared) from the Kaggle session
```

- Serving the UI from the same FastAPI app keeps one URL and avoids cross-origin problems. FastAPI middleware handles validation, logging and CORS.
- **Kaggle is not a normal web host.** The tunnel URL is temporary and changes on every restart. A third-party guide puts the session limit at about 12 hours (not confirmed from Kaggle's own documentation). The tunnel token goes in Kaggle Secrets, and notebook internet access must be on. Assume the SQLite file resets with the session: save it as output and reload it at startup.
- Start the session shortly before evaluation, test the URL from another device, and keep backups: the public notebook link, the saved model and a screen recording of the demo.

## 11. Plan for the remaining days

| Day | Goal |
|---|---|
| Sun 15 Nov | Run 311 clustering. Send the instructor the questions in section 2. |
| Mon 16 | Test cluster-as-feature on validation weeks. Freeze final metrics and the priority table. |
| Tue 17 | Write FastAPI endpoints and the SQLite CRUD layer. |
| Wed 18 | Connect the website to the API with fetch. Add the saved-scenarios control page. |
| Thu 19 | First Kaggle plus tunnel hosting test from another device. |
| Fri 20 | Full rehearsal, backups, report and viva answers. |
| Sat 21 | Start a fresh session, test the URL, submit. |

## 12. Open items and claims we must not make

- **Open:** 311 clustering; FastAPI; SQL CRUD; hosting test; instructor answers; optionally a longer export (several months, excluding NYPD, which is 41 percent of rows and never slow) because 3.5 weeks cannot show seasons or explain the drift.
- **Do not claim:** that the model decides urgency; that acting on its flags speeds things up; that it replaces routing; that a quiet ZIP is fine (people report to 311 at different rates); that explanations show causes (they show associations); that the 311 and air-quality data can share one map (different cities and years).
- **Unverified:** why the slow share dropped around 26 Oct; whether complaint channel, repeat addresses or resolution text help (not tested); Kaggle's real session limits; whether the Google Air Quality API covers the student's region.

## 13. Glossary

| Term | Meaning |
|---|---|
| X and y | X is what the model may look at; y is the answer it predicts. Here y is slow_48h, which we build from the created and closed dates. |
| Supervised / unsupervised | Supervised learns from a y column (our prediction task). Unsupervised has no y and finds groups on its own (clustering). The brief requires both. |
| Train / validation / test | Learn here / choose and calibrate here / final unseen exam. Split by date so the test is the future. |
| Leakage | Letting information that would not be known at prediction time into X. It makes scores falsely perfect. |
| PR-AUC, ROC-AUC | Ranking scores for imbalanced yes/no problems. Higher is better. Compare with the baseline, not with 1.0. |
| Brier score, calibration | Brier measures probability error (lower is better). Calibrated means 30 percent predicted really happens about 30 percent of the time. |
| Drift | The situation changes over time, so a model trained earlier becomes less reliable. Here: the slow share fell sharply. |
| Slow share | Percentage of complaints that were slow (not closed within 48 hours). |
| Silhouette, bootstrap stability | Silhouette measures how well separated clusters are. Stability re-clusters resampled data to see whether the same groups appear. |

## 14. Likely viva questions, short answers

- **Why a model if departments already handle complaints?** 311 routes them already. The delay is inside long queues. We flag likely-stuck complaints early and give the city one explainable view for splitting limited crews. The gain is modest and we say so.
- **Does it make things faster?** We cannot prove that from this data. We show the flags carry real signal on unseen weeks and a department would have to test the effect.
- **Why only 3 complaint types?** On all types the agency decides the outcome, so a lookup table matches the model. We test where outcomes genuinely vary, chosen from training weeks only.
- **Is 0.51 good?** It beats the 0.40 baseline modestly, on unseen later weeks, under a sharp drift. We report it as a modest real signal, not a breakthrough.
- **Why both supervised and unsupervised?** The brief requires both. Prediction says how likely a complaint is to stall; clustering says what kind of situation it is. The priority score and the explanation panel use both.
- **How is urgency handled?** It is not learned. Impact weights are visible assumptions that city staff can change.

## 15. Starter message for the new AI

Paste this in the new chat together with the re-attached files:

```
I am a student on a team of 4, due 21 Nov 2026. I attached a project handoff PDF and my files.
Read the PDF fully first. Project: 'Integrated Urban Risk Analytics and Resource
Prioritization System', main data NYC 311, second UCI air quality.
Rules: simple English, short answers, one question at a time, never invent numbers,
say what you could not verify, and keep our honest results (311 model PR-AUC 0.51 vs
0.40 baseline, heavy drift). Next job: step 1 below, then tell me before starting each
next step.
1. Run 311 clustering (profiles by ZIP, complaint type, time of day) with a stability
   check, fitted on the training period only.
2. Write the FastAPI app (predict, priority, clusters, metrics, CRUD scenarios in SQLite).
3. Connect my website (urban-risk-desk.html) to it with fetch.
4. Make the Kaggle notebook that serves the API through a tunnel.
5. Prepare the report and viva answers.
```


## 16. Files in text form (this chat only accepts images and text files)

Some files were renamed so they can be attached. Tell the student to attach these:

- `pipeline_311_py.txt` is `pipeline_311.py` (rename it back to .py before running).
- `urban-risk-desk_html.txt` is `urban-risk-desk.html` (rename it back to .html to open it).
- `nyc311_sample_5000.csv` is a random 5,000-row sample of the cleaned data (same columns as `nyc311_clean.csv`). The full 41 MB file may be too big to attach. For real runs the student should use the full `nyc311_clean.csv` or `export.csv` on their own computer or Kaggle.
- `results.json` can be attached as it is.
- Air-quality data: the student can paste the first rows of `AirQualityUCI.csv` as a text file. It is semicolon separated with decimal commas and -200 for missing values.
