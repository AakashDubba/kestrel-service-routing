# Kestrel Home Appliances — Service Request Routing

## Overview

Kestrel Home Appliances uses an external vendor bot to route incoming customer service requests to internal support teams. This project audits the bot's historical accuracy against actual resolution outcomes and builds a replacement text classifier trained on the team that actually resolved each ticket (`final_team`), rather than the team the bot originally assigned (`team_label`).

## Business Finding

An audit of 10,822 historically resolved service requests produced the following findings:

| Metric | Value |
|---|---|
| Tickets analysed | 10,822 |
| Legacy bot accuracy | 77.17% |
| Legacy bot error rate | 22.83% |
| Misroutes | 2,471 |
| Cost per misroute | Rs 565 (Rs 305 transfer + Rs 260 extra contact) |
| Historical misrouting cost identified | Rs 13,96,115 |
| Annual vendor bot licence | Rs 3,20,000 |

The Rs 13,96,115 figure represents the historical misrouting cost identified from the legacy bot's observed errors across the audit dataset, calculated at Rs 565 per misrouted ticket.

## Modeling Approach

- **Target:** `final_team` — the team that actually resolved each ticket. `team_label` (the bot's original assignment) is used only for the legacy comparison audit.
- **Features:** Character-level n-gram TF-IDF from customer request text (`analyzer='char_wb'`, `ngram_range=(3, 6)`, `sublinear_tf=True`).
- **Classifier:** `LinearSVC(C=0.1, random_state=42)`.
- **Team normalization:** Legacy names "Installations" and "Consumables" are mapped to their current equivalents "Installs & Demo" and "Filters & Consumables" per operations policy Section 5.
- The model runs locally with no external API calls. Direct model inference cost: Rs 0.00 per prediction.

## Validation

All performance metrics are from a fixed stratified 80/20 train/validation split (`random_state=42`):

| Metric | Value |
|---|---|
| Validation records | 2,165 |
| Accuracy | 84.71% |
| Macro F1 | 0.8557 |
| Weighted F1 | 0.8492 |
| Validation errors | 331 |

- The 20% holdout was not used during model selection or tuning.
- `test_unlabelled.csv.csv` contains no target labels and was used only for generating the final submission predictions in `predictions.csv`.

## API

The FastAPI application (`app.py`) exposes the following endpoints:

### Health Check
```http
GET /health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "valid_teams": [
    "Billing",
    "Filters & Consumables",
    "Installs & Demo",
    "Product Advice",
    "Repairs",
    "Returns & Replacement",
    "Warranty Claims"
  ]
}
```

### Predict
```http
POST /predict
Content-Type: application/json

{
  "text": "Water purifier is leaking from the bottom"
}
```
Response:
```json
{
  "team": "Repairs",
  "reason": "This request was routed to Repairs based on patterns learned from 10,822 historically resolved service requests. Product faults, breakdowns, error codes, noise, leaks — anything needing a technician."
}
```

## Frontend

The root endpoint (`GET /`) serves a single-page triage interface (`index.html`) where a support agent can paste a customer message, submit it, and see the predicted team and routing reason.

## Cost

- Direct model inference cost: **Rs 0.00 per prediction** under local deployment.
- No paid external inference API is used.

## Limitations

- The model achieves 84.71% holdout accuracy, meaning approximately 15% of tickets may still require manual triage or reassignment.
- There is no automated drift monitoring or automated retraining pipeline.
- The model uses only the customer's request text, without structured fields such as product family or warranty status.
- Future appliance categories or new support teams would require retraining.

## Repository Contents

| File | Description |
|---|---|
| `.gitignore` | Exclusion rules for raw data, caches, and local-only files. |
| `README.md` | Project documentation and evaluation report. |
| `app.py` | FastAPI application exposing `/health` and `/predict` endpoints and serving the frontend. |
| `index.html` | Single-page triage interface for support agents. |
| `memo.md` | Business memorandum summarising the audit findings and deployment plan. |
| `predictions.csv` | Predictions for the 2,178 test requests (`request_id`, `team`). |
| `requirements.txt` | Python dependencies. |
| `routing_model.pkl` | Serialized scikit-learn pipeline (TF-IDF + LinearSVC). |
| `train_and_predict.py` | Training, validation, model serialization, and test prediction generation. |

## How to Run

```bash
pip install -r requirements.txt
python train_and_predict.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open: http://localhost:8000
