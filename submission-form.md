# Banao Technologies — Task 2 Submission Form

## 1. What did you try?

I built a complete service-request routing system for Kestrel Home Appliances. The core steps were:

- **Data join:** Joined `train.csv` (10,822 service requests with the vendor bot's `team_label`) with `resolution_log.csv` (which records `final_team` — the team that actually resolved each request). This was an inner join on `request_id` with zero data loss: all 10,822 rows matched perfectly, no duplicates, no nulls.

- **Team name normalization:** Applied the ops-policy.pdf Section 5 rules — "Installations" became "Installs & Demo" and "Consumables" became "Filters & Consumables" — reducing 9 raw team labels to 7 canonical teams.

- **Old bot misrouting analysis:** Compared the vendor bot's `team_label` against `final_team` (both normalized). Found 2,471 misroutes out of 10,822 tickets (22.83% error rate). At Rs 565 per misroute (Rs 305 transfer cost + Rs 260 extra contact cost, from ops-policy.pdf Section 4), the total historical misrouting cost was Rs 13,96,115.

- **Model training:** Built a TF-IDF (character n-grams) + LinearSVC pipeline predicting `final_team` (the resolution target, not the bot's label). Validated on a stratified 80/20 split: 84.71% accuracy, 0.8557 macro F1, 0.8492 weighted F1. Then retrained on all 10,822 rows and saved as `routing_model.pkl`.

- **Test predictions:** Generated predictions for all 2,178 test requests. Output matches the `sample_submission.csv` format exactly (request_id, team), all teams are valid, no missing predictions.

- **API and frontend:** Built a FastAPI service (`app.py`) that loads the model at startup, exposes `/health` and `/predict` endpoints, and serves a professional dark-themed frontend (`index.html`).

## 2. What did you change?

**The training target.** The task brief asks for ~90% agreement with the old routing bot. I deliberately trained against `final_team` (the team that actually closed each ticket) instead of `team_label` (the bot's original assignment). This is the critical change — the old bot is wrong 22.83% of the time, so copying it would reproduce Rs 13,96,115 in misrouting waste.

## 3. What did you discard?

- **Matching the old bot:** The 90% agreement target was a trap. The bot misroutes nearly one in four tickets. Optimizing for agreement would mean training the model to be wrong in the same ways the bot is wrong.
- **Complex feature engineering:** I considered adding channel, product_family, warranty_status, and source as features, but the text-only pipeline already achieves 84.71% accuracy with a simple, interpretable architecture. Adding structured features would add engineering complexity with marginal gains and risk overfitting.
- **Deep learning / LLMs:** Not needed for this problem size. A TF-IDF + LinearSVC pipeline is fast, interpretable, runs offline, costs nothing, and performs well.

## 4. What did you deliberately NOT build?

- **An LLM-based classifier** — would add API cost, latency, and a vendor dependency, defeating the purpose of replacing the paid bot.
- **A retraining pipeline** — premature for a first deployment. Once the model is running in production, we can evaluate drift and build retraining if needed.
- **A dashboard** — the memo communicates the key metrics. A live dashboard can follow once the model is in production.

## 5. What did you build that was not explicitly requested?

- **Detailed error analysis** in the training script showing the top misclassification patterns (e.g., "Installs & Demo" confused with "Repairs"), with sample misclassified texts.
- **Team descriptions in the API response** — the `/predict` endpoint returns a `reason` field explaining which team was selected and what it handles, making the routing decision transparent to agents.
- **Old bot misroute pattern breakdown** — the script prints the top 10 routing errors by (bot_team → actual_team), revealing that the bot systematically confuses "Billing" and "Filters & Consumables" requests with "Repairs."

## 6. What are the three most important things another engineer needs to know if they take over on Monday?

1. **The model predicts `final_team`, not `team_label`.** The entire value proposition depends on this. The old bot's `team_label` is wrong 22.83% of the time. If anyone retrains against `team_label`, they will reproduce those errors and the Rs 565-per-misroute cost.

2. **Team names must be normalized before any comparison or training.** The ops-policy.pdf (Section 5, effective 15 Jan 2026) renamed "Installations" to "Installs & Demo" and "Consumables" to "Filters & Consumables." The raw data contains both old and new names. Without normalization, the model sees 9 classes instead of 7 and comparisons are silently wrong.

3. **The test set (`test_unlabelled.csv`) was never used for training or validation.** The 84.71% accuracy figure comes from a held-out 20% stratified split of the joined training data. The test predictions in `predictions.csv` are the model's first and only pass over unseen data.

## 7. How many hours did the task actually take?

[USER TO FILL]

## 8. Key metrics (verified from code execution)

| Metric | Value |
|---|---|
| Training rows | 10,822 |
| Validation rows | 2,165 |
| Final classes | 7 (Billing, Filters & Consumables, Installs & Demo, Product Advice, Repairs, Returns & Replacement, Warranty Claims) |
| Old bot comparable tickets | 10,822 |
| Old bot correct | 8,351 |
| Old bot misroutes | 2,471 |
| Old bot error rate | 22.83% |
| Old bot accuracy | 77.17% |
| Cost per misroute | Rs 565 |
| Total historical misrouting cost | Rs 13,96,115 |
| New model validation accuracy | 84.71% |
| Macro F1 | 0.8557 |
| Weighted F1 | 0.8492 |
| Test predictions | 2,178 rows |
| Prediction cost | Rs 0.00 per prediction |
| Monthly model cost | Rs 0.00 |
| Annual bot licence saved | Rs 3,20,000 |

## 9. Misroute cost discovery — why we pushed back on 90% match

The task brief said: achieve approximately 90% agreement with the old routing bot. This sounds reasonable until you check the evidence.

The vendor bot misroutes 2,471 out of 10,822 tickets — a 22.83% error rate. Each misroute costs Rs 565 (Rs 305 in agent transfer time + Rs 260 in extra customer contacts, per ops-policy.pdf Section 4). That is Rs 13,96,115 in hidden costs sitting in the historical data.

If we had trained a model to match the bot at 90%+, we would be training it to replicate those errors. The model would learn that "customer paid online but item not delivered" is a Billing problem (the bot's label), when it actually needs Returns & Replacement (where it was resolved). Optimizing for agreement with a wrong system perpetuates the waste.

Instead, we trained against `final_team` — the team that actually closed each ticket — so the new model learns correct routing from real outcomes. The result is an 84.71% accurate model that costs Rs 0 to run and does not reproduce the bot's systematic errors.

## 10. Repository / submission links

- **GitHub repo:** https://github.com/AakashDubba/kestrel-service-routing
- **Files included:** `train_and_predict.py`, `app.py`, `index.html`, `routing_model.pkl`, `predictions.csv`, `memo.md`, `submission-form.md`, `requirements.txt`

## 11. How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Train model and generate predictions
python train_and_predict.py

# Start the API server
uvicorn app:app --host 0.0.0.0 --port 8000

# Open browser to http://localhost:8000
```
