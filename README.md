# Kestrel Home Appliances — Service Request Routing

## Business Problem

Kestrel Home Appliances previously relied on an external legacy vendor bot to triage incoming customer service requests across internal support and engineering teams. An empirical audit of historical service interactions was conducted by joining the intake ticketing records with actual resolution logs (`final_team`).

The audit revealed that the legacy bot frequently misrouted tickets, introducing costly triage delays, unnecessary cross-team reassignments, and increased customer friction.

## Key Finding

Auditing the historical ticketing data against actual resolution outcomes established the following verified baseline metrics for the legacy bot:

- **Old bot accuracy:** 77.17%
- **Old bot error rate:** 22.83%
- **Misroutes:** 2,471 tickets
- **Cost per misroute:** Rs 565 (comprising Rs 305 internal department transfer cost + Rs 260 extra customer contact friction per Ops Policy Section 4)
- **Historical misrouting cost:** Rs 13,96,115 (across 2,471 misrouted requests in the audit dataset)

Crucially, attempting to reproduce or align with the legacy bot's predictions (`team_label`) would merely codify its 22.83% error rate and perpetuate over Rs 13.9 Lakhs in operational waste. Consequently, the replacement machine learning model was trained strictly on ground-truth resolution outcomes (`final_team`), teaching the model where requests are actually solved rather than copying flawed legacy decisions.

## Final Model

The production model is a robust, character-level text classification pipeline designed to handle product codes, typos, informal phrasing, and colloquial terms:

- **Vectorization:** Character n-gram TF-IDF (`analyzer='char_wb'`, `ngram_range=(3, 6)`, `sublinear_tf=True`)
- **Classifier:** `LinearSVC(C=0.1, random_state=42)`
- **Holdout Accuracy:** **84.71%** (evaluated against true `final_team` on an untouched 20% stratified holdout split)
- **Macro F1 Score:** **0.8557**
- **Weighted F1 Score:** **0.8492**

### Distinguishing True-Resolution Accuracy vs. Legacy Agreement

It is vital to distinguish between two fundamentally different evaluation metrics:
1. **True-Resolution Accuracy (84.71%):** The rate at which the new model correctly predicts the actual resolving team (`final_team`). This represents genuine operational routing efficacy (+7.54 percentage points over the old bot's 77.17%).
2. **Agreement with Legacy Bot (82.15%):** The rate at which the new model agrees with the legacy bot's original tags (`team_label`). Because the old bot was flawed on nearly 23% of cases, higher agreement with the old bot is an anti-goal that harms customer resolution.

## Architecture

The end-to-end inference flow operates locally and deterministically:

```
Customer Request (Text)
       │
       ▼
Character TF-IDF Representation (char_wb, n-grams 3–6)
       │
       ▼
Linear Support Vector Classifier (LinearSVC, C=0.1)
       │
       ▼
Recommended Team (+ Decision Confidence & Latency)
```

## API

The service is packaged using FastAPI (`app.py`), providing production-ready endpoints for automated systems and internal agent dashboards.

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": [
    "Commercial & Bulk",
    "Customer Care",
    "Electrical Escalations",
    "Filters & Consumables",
    "Installs & Demo",
    "Mechanical & Motor Support",
    "Returns & Replacements",
    "Smart Appliance IoT Support"
  ]
}
```

### 2. Service Request Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "Water purifier is leaking from the bottom valve and filter light is blinking red"
}
```
**Response:**
```json
{
  "request_text": "Water purifier is leaking from the bottom valve and filter light is blinking red",
  "predicted_team": "Filters & Consumables",
  "confidence": 0.884,
  "latency_ms": 1.2
}
```

Another example:
```http
POST /predict
Content-Type: application/json

{
  "text": "App not syncing with chimney over home wifi, shows offline status"
}
```
**Response:**
```json
{
  "request_text": "App not syncing with chimney over home wifi, shows offline status",
  "predicted_team": "Smart Appliance IoT Support",
  "confidence": 0.912,
  "latency_ms": 1.1
}
```

## How to Run

### 1. Environment Setup
Clone the repository and install the minimal dependencies:
```bash
python -m pip install -r requirements.txt
```

### 2. Train and Generate Predictions
Run the end-to-end training, holdout validation, and batch inference script:
```bash
python train_and_predict.py
```
This script validates the model on a fixed 20% stratified holdout, trains the final production pipeline on all available verified data, serializes `routing_model.pkl`, and exports test set inferences to `predictions.csv`.

### 3. Launch API & Interactive Interface
Launch the FastAPI server:
```bash
uvicorn app:app --reload
```
Then navigate to:
```
http://localhost:8000
```
The root page serves an agent-facing triage dashboard (`index.html`) featuring live single-request triage, batch processing, latency profiling, and system health status.

## Repository Contents

- `README.md`: Executive summary, architecture, API documentation, and reproduction instructions.
- `train_and_predict.py`: Core pipeline script containing data ingestion, team normalization, holdout validation, model serialization, and test prediction generation.
- `app.py`: High-performance FastAPI application exposing `/health` and `/predict` endpoints, static dashboard mounting, and CORS middleware.
- `index.html`: Responsive, dark-mode web dashboard for support agents with real-time prediction and confidence metrics.
- `requirements.txt`: Pinned, minimal production dependencies (`pandas`, `numpy`, `scikit-learn`, `joblib`, `fastapi`, `uvicorn`).
- `predictions.csv`: Model predictions for all entries in `test_unlabelled.csv.csv`, formatted with `request_id` and `team`.
- `routing_model.pkl`: Serialized, self-contained scikit-learn pipeline (TF-IDF vectorizer + LinearSVC classifier).
- `memo.md`: Executive memorandum addressed to operations and finance leadership detailing financial findings, bot audit, and deployment plan.
- `submission-form.md`: Comprehensive engineering questionnaire covering design decisions, model selection, failure analysis, and cost trade-offs.
- `optimize_model.py`: Benchmarking script evaluating 6 distinct model families and hyperparameter configurations across stratified cross-validation.
- `.gitignore`: Standard exclusion rules for Python caches, virtual environments, IDE metadata, and local assessment datasets.

## Business Impact

- **Rejecting Legacy Bot Mimicry:** The legacy bot operated with an audited 22.83% failure rate. Optimizing to mimic its decisions would lock in bad routing habits. Training on verified resolution outcomes (`final_team`) directly targets actual operational success.
- **Eliminating Historical Misrouting Waste:** By improving routing accuracy from 77.17% to 84.71% (+7.54 percentage points), the new solution prevents hundreds of misroutes annually, directly mitigating the Rs 565 cost incurred per misrouted request.
- **Rs 0.00 Prediction Software & API Cost:** The system uses scikit-learn running locally on existing infrastructure with sub-2ms latency. It requires no external paid LLM API calls, no token-based billing, and eliminates the legacy vendor's recurring licensing fees.
- **Air-Gapped & Offline Capable:** Entirely self-contained pipeline; customer request data never leaves the internal network, ensuring total data privacy.

## Validation

All reported performance metrics were evaluated using a strict, fixed stratified 80/20 train/test split (`random_state=42`) on the labeled dataset:
- Hyperparameter tuning and model family comparisons were conducted exclusively within the 80% training split using 5-fold stratified cross-validation.
- The 20% holdout set remained completely untouched until final evaluation to ensure zero data leakage.
- **Unlabelled Test Set Notice:** The `test_unlabelled.csv.csv` file contains no target labels and was **not** used for model evaluation or metric computation. It was strictly used for generating the final submission batch predictions in `predictions.csv`.

## Limitations

- **Performance Scope:** An 84.71% holdout accuracy means approximately 15.3% of tickets will still require manual triage or reassignment. The model is an automated first-line triage system, not an infallible oracle.
- **Domain Drift:** Future appliances or newly created support departments will require incremental retraining as language patterns shift.
- **Text-Only Context:** The model operates solely on the incoming ticket text without customer purchase history or hardware telemetry. While intentionally chosen for zero-dependency intake, integrating validated CRM metadata in future iterations may yield further performance improvements.
