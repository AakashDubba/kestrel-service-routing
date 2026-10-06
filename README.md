# Kestrel Home Appliances - Service Routing Model

## Business Problem
Kestrel Home Appliances currently relies on an external vendor bot to route customer service requests. An audit of historical resolutions revealed that the bot's initial classification (`team_label`) was incorrect for **22.83%** of tickets when compared against the team that actually resolved the ticket (`final_team`). 

Every misrouted ticket costs Kestrel Rs 565 in internal transfer time (Rs 305) and additional customer contact delays (Rs 260). With 2,471 misroutes in the training data alone, the bot's errors represent a hidden historical cost of **Rs 13,96,115**. 

## The Solution
Instead of attempting to reach 90% agreement with an inaccurate bot (which would simply perpetuate the Rs 13.9L waste), this project replaces the bot entirely. We trained a machine learning model to predict `final_team`—the correct historical resolution outcome. 

**Key Achievements:**
- **Eliminates Vendor Fees:** Removes the Rs 3.2L annual vendor bot license.
- **Costs Rs 0 to Run:** Completely local, offline text classification pipeline.
- **Significantly More Accurate:** The final model achieves **84.71% validation accuracy** against actual resolution outcomes, outperforming the old bot's **77.17%** accuracy by a solid 7.54 percentage points.

## Architecture & Final Model
The system is built as a lightweight NLP pipeline in Python:
- **Preprocessing:** Normalizes legacy team names to their modern equivalents (e.g., "Installations" -> "Installs & Demo") based on operations policy.
- **Feature Extraction:** Extracts Character N-Grams (3 to 6 characters, sublinear TF-IDF scaling) from the raw customer request text.
- **Classifier:** A `LinearSVC` (Support Vector Classifier) with `C=0.1`.
- **API Backend:** A FastAPI endpoint that loads the `.pkl` model and serves predictions instantly.
- **Frontend:** A clean, Vanilla HTML/CSS interface mimicking Kestrel's internal agent dashboard.

## File Structure
- `train_and_predict.py`: The core pipeline script that joins training data, evaluates old bot metrics, validates the new model using a stratified 80/20 split, trains on the full valid dataset, and outputs predictions for the unlabelled test set.
- `app.py`: FastAPI server for the routing model.
- `index.html`: The frontend user interface for testing live predictions.
- `routing_model.pkl`: The serialized, production-ready scikit-learn model.
- `predictions.csv`: The final predictions generated for `test_unlabelled.csv`.
- `memo.md`: A business-facing summary of the financial findings and deployment plan.
- `submission-form.md`: Banao task submission responses detailing engineering decisions.

## Installation & Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model and generate predictions:
   ```bash
   python train_and_predict.py
   ```

3. Start the API server:
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. Access the web interface at `http://localhost:8000` or use the API:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "my water purifier is leaking from the bottom"}'
   ```

## Limitations & Handover Notes
- **Target Variable Importance:** The model was trained against `final_team` (outcomes), not `team_label` (the bot). Any future retraining **must** continue using `final_team` to avoid regressing to the vendor bot's poor performance.
- **Normalization Required:** Legacy data contains old team names. The normalization step (defined in `train_and_predict.py` according to operations policy) must run before training.
- **No Structured Features Used:** We deliberately excluded categorical fields (like `product_family`) to keep the frontend completely generic (text-only). Future iterations can include them if the data collection pipeline guarantees those fields are populated at intake.
