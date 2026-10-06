"""
Kestrel Home Appliances — Service Request Routing Model
========================================================
train_and_predict.py

This script:
1. Loads and joins train.csv.csv with resolution_log.csv.csv
2. Normalizes team names per ops-policy.pdf Section 5
3. Calculates old bot misrouting cost and error rate from actual data
4. Validates the new model on a stratified 80/20 split
5. Trains the final model on all valid data
6. Saves routing_model.pkl
7. Predicts on test_unlabelled.csv.csv and outputs predictions.csv
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import os
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv.csv")
RESOLUTION_FILE = os.path.join(DATA_DIR, "resolution_log.csv.csv")
TEAMS_FILE = os.path.join(DATA_DIR, "teams.csv.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_unlabelled.csv.csv")
SAMPLE_SUB_FILE = os.path.join(DATA_DIR, "sample_submission.csv.csv")
MODEL_FILE = os.path.join(DATA_DIR, "routing_model.pkl")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.csv")

# Cost constants from ops-policy.pdf Section 4
TRANSFER_COST_RS = 305  # Rs per transfer
EXTRA_CONTACT_COST_RS = 260  # Rs per additional customer contact from misroute
COST_PER_MISROUTE = TRANSFER_COST_RS + EXTRA_CONTACT_COST_RS  # Rs 565

# Team name normalization map from ops-policy.pdf Section 5
# "From 15 Jan 2026 the Installations team became Installs & Demo,
#  and Consumables became Filters & Consumables."
TEAM_RENAME_MAP = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}


def normalize_team(name):
    """Normalize team name using the ops-policy.pdf Section 5 rules."""
    return TEAM_RENAME_MAP.get(name, name)


# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

train = pd.read_csv(TRAIN_FILE)
resolution = pd.read_csv(RESOLUTION_FILE)
teams = pd.read_csv(TEAMS_FILE)
test = pd.read_csv(TEST_FILE)
sample_sub = pd.read_csv(SAMPLE_SUB_FILE)

print(f"Train rows:            {len(train)}")
print(f"Resolution log rows:   {len(resolution)}")
print(f"Test rows:             {len(test)}")
print(f"Sample submission rows: {len(sample_sub)}")

# ──────────────────────────────────────────────
# 2. Join train with resolution_log
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Joining train with resolution_log")
print("=" * 60)

# Check for duplicates
train_dup = train.request_id.duplicated().sum()
res_dup = resolution.request_id.duplicated().sum()
print(f"Duplicate request_ids in train:      {train_dup}")
print(f"Duplicate request_ids in resolution: {res_dup}")

# Inner join on request_id
df = train.merge(resolution, on="request_id", how="inner")
print(f"\nTraining rows before join: {len(train)}")
print(f"Training rows after join:  {len(df)}")
print(f"Rows lost in join:         {len(train) - len(df)}")

# Check for missing final_team
missing_final = df.final_team.isnull().sum()
print(f"Rows with missing final_team: {missing_final}")

# Remove rows with missing final_team or request_text
missing_text = df.request_text.isnull().sum()
print(f"Rows with missing request_text: {missing_text}")

rows_before_clean = len(df)
df = df.dropna(subset=["final_team", "request_text"])
df = df[df.request_text.str.strip() != ""]
rows_after_clean = len(df)
removed = rows_before_clean - rows_after_clean
if removed > 0:
    print(f"Records removed (missing text/team): {removed}")
else:
    print("Records removed: 0 (all rows valid)")

# ──────────────────────────────────────────────
# 3. Normalize team names
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Normalizing team names (ops-policy.pdf Section 5)")
print("=" * 60)

# Show raw values first
print(f"\nRaw final_team unique values ({df.final_team.nunique()}):")
for t in sorted(df.final_team.unique()):
    print(f"  - {t}")

print(f"\nRaw team_label unique values ({df.team_label.nunique()}):")
for t in sorted(df.team_label.unique()):
    print(f"  - {t}")

# Apply normalization
df["final_team_norm"] = df["final_team"].apply(normalize_team)
df["team_label_norm"] = df["team_label"].apply(normalize_team)

print(f"\nNormalized final_team unique values ({df.final_team_norm.nunique()}):")
for t in sorted(df.final_team_norm.unique()):
    print(f"  - {t}")

print(f"\nNormalized team_label unique values ({df.team_label_norm.nunique()}):")
for t in sorted(df.team_label_norm.unique()):
    print(f"  - {t}")

# ──────────────────────────────────────────────
# 4. Final class distribution
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Final class distribution (target = final_team_norm)")
print("=" * 60)

class_dist = df.final_team_norm.value_counts().sort_index()
for team, count in class_dist.items():
    pct = 100 * count / len(df)
    print(f"  {team:30s}  {count:5d}  ({pct:5.1f}%)")
print(f"  {'TOTAL':30s}  {len(df):5d}")

# ──────────────────────────────────────────────
# 5. Old Bot Evidence — Misrouting Analysis
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Old bot misrouting analysis")
print("=" * 60)

total_comparable = len(df)
correct = (df.team_label_norm == df.final_team_norm).sum()
misroutes = total_comparable - correct
error_rate = misroutes / total_comparable
accuracy_old = correct / total_comparable
total_misrouting_cost = misroutes * COST_PER_MISROUTE

print(f"\nTotal comparable tickets:    {total_comparable}")
print(f"Old bot correct routings:    {correct}")
print(f"Old bot misroutes:           {misroutes}")
print(f"Old bot error rate:          {error_rate:.4f} ({error_rate*100:.2f}%)")
print(f"Old bot accuracy:            {accuracy_old:.4f} ({accuracy_old*100:.2f}%)")
print(f"Cost per misroute:           Rs {COST_PER_MISROUTE}")
print(f"Total misrouting cost:       Rs {total_misrouting_cost:,.0f}")
print(f"                             Rs {total_misrouting_cost/100000:.2f} lakh")

# Show where the old bot goes wrong (top misroute patterns)
print("\nTop 10 misroute patterns (old bot team -> actual team):")
misrouted_df = df[df.team_label_norm != df.final_team_norm]
misroute_patterns = (
    misrouted_df.groupby(["team_label_norm", "final_team_norm"])
    .size()
    .sort_values(ascending=False)
    .head(10)
)
for (old, actual), count in misroute_patterns.items():
    print(f"  {old:30s} -> {actual:30s}  ({count} tickets)")

# ──────────────────────────────────────────────
# 6. Model Training — Validation Split
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Model validation (80/20 stratified split)")
print("=" * 60)

X = df["request_text"]
y = df["final_team_norm"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training split:   {len(X_train)} rows")
print(f"Validation split: {len(X_val)} rows")

# Build pipeline
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True)),
        ("clf", LinearSVC(dual="auto", C=0.1, random_state=42)),
    ]
)

# Train on split
pipeline.fit(X_train, y_train)
y_pred_val = pipeline.predict(X_val)

# Metrics
val_accuracy = accuracy_score(y_val, y_pred_val)
val_macro_f1 = f1_score(y_val, y_pred_val, average="macro")
val_weighted_f1 = f1_score(y_val, y_pred_val, average="weighted")

print(f"\nValidation Accuracy:    {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Validation Macro F1:    {val_macro_f1:.4f}")
print(f"Validation Weighted F1: {val_weighted_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred_val))

print("Confusion Matrix:")
labels = sorted(y.unique())
cm = confusion_matrix(y_val, y_pred_val, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# ──────────────────────────────────────────────
# 7. Error Analysis
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Error analysis — representative incorrect predictions")
print("=" * 60)

val_results = pd.DataFrame(
    {
        "request_text": X_val.values,
        "true_team": y_val.values,
        "predicted_team": y_pred_val,
    }
)
errors = val_results[val_results.true_team != val_results.predicted_team]
print(f"Total validation errors: {len(errors)} / {len(y_val)}")

# Show error pattern summary
error_patterns = (
    errors.groupby(["predicted_team", "true_team"])
    .size()
    .sort_values(ascending=False)
    .head(8)
)
print("\nTop error patterns (predicted -> true):")
for (pred, true), count in error_patterns.items():
    print(f"  {pred:30s} -> {true:30s}  ({count} cases)")

# Show a few example errors
print("\nSample misclassified requests:")
for i, row in errors.head(5).iterrows():
    text_preview = row["request_text"][:100] + "..." if len(row["request_text"]) > 100 else row["request_text"]
    print(f"  Text: {text_preview}")
    print(f"  Predicted: {row['predicted_team']}  |  Actual: {row['true_team']}")
    print()

# ──────────────────────────────────────────────
# 8. Final Model — Train on ALL data
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Training final model on ALL valid data")
print("=" * 60)

final_pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True)),
        ("clf", LinearSVC(dual="auto", C=0.1, random_state=42)),
    ]
)

final_pipeline.fit(X, y)
print(f"Final model trained on {len(X)} rows")

# Save model
joblib.dump(final_pipeline, MODEL_FILE)
print(f"Model saved to: {MODEL_FILE}")

# ──────────────────────────────────────────────
# 9. Predict on test_unlabelled.csv.csv
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Predicting on test_unlabelled.csv.csv")
print("=" * 60)

test_predictions = final_pipeline.predict(test["request_text"])

# Build submission dataframe
submission = pd.DataFrame(
    {"request_id": test["request_id"], "team": test_predictions}
)

# Validate predictions
allowed_teams = set(y.unique())
predicted_teams = set(submission["team"].unique())
print(f"Allowed teams:   {sorted(allowed_teams)}")
print(f"Predicted teams: {sorted(predicted_teams)}")

invalid_teams = predicted_teams - allowed_teams
if invalid_teams:
    print(f"WARNING: Invalid teams found: {invalid_teams}")
else:
    print("All predicted teams are valid.")

missing_preds = submission["team"].isnull().sum()
print(f"Missing predictions: {missing_preds}")

# Check against sample submission format
print(f"\nSubmission shape:        {submission.shape}")
print(f"Sample submission shape: {sample_sub.shape}")
print(f"Columns match: {list(submission.columns) == list(sample_sub.columns)}")

# Save
submission.to_csv(PREDICTIONS_FILE, index=False)
print(f"Predictions saved to: {PREDICTIONS_FILE}")

# ──────────────────────────────────────────────
# 10. Summary
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Training data:            {len(df)} rows")
print(f"Validation split:         {len(X_val)} rows")
print(f"Old bot accuracy:         {accuracy_old*100:.2f}%")
print(f"Old bot misroutes:        {misroutes}")
print(f"Old bot misrouting cost:  Rs {total_misrouting_cost:,.0f}")
print(f"New model val accuracy:   {val_accuracy*100:.2f}%")
print(f"New model val macro F1:   {val_macro_f1:.4f}")
print(f"New model val weighted F1:{val_weighted_f1:.4f}")
print(f"Test predictions:         {len(submission)} rows")
print(f"Model file:               {MODEL_FILE}")
print(f"Predictions file:         {PREDICTIONS_FILE}")

# Prediction distribution
print("\nPrediction distribution on test set:")
pred_dist = submission.team.value_counts().sort_index()
for team, count in pred_dist.items():
    pct = 100 * count / len(submission)
    print(f"  {team:30s}  {count:5d}  ({pct:5.1f}%)")
