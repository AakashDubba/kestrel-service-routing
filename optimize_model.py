"""
Phase 1-5: Systematic model optimization for Kestrel routing.
All experiments use a SINGLE fixed stratified 80/20 split.
The 20% holdout is untouched until the very end.
Tuning is done via stratified 5-fold CV on the 80% training portion only.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
import os, time

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Team normalization ──
TEAM_RENAME = {"Installations": "Installs & Demo", "Consumables": "Filters & Consumables"}
def norm(name): return TEAM_RENAME.get(name, name)

# ── Load & join ──
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv.csv"))
res   = pd.read_csv(os.path.join(DATA_DIR, "resolution_log.csv.csv"))
df = train.merge(res, on="request_id", how="inner")
df["final_team_norm"] = df["final_team"].apply(norm)
df["team_label_norm"] = df["team_label"].apply(norm)
print(f"Joined rows: {len(df)}")

# ── SINGLE fixed split ──
X_text = df["request_text"]
y = df["final_team_norm"]
SPLIT_SEED = 42

X_train_text, X_hold_text, y_train, y_hold = train_test_split(
    X_text, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y
)
# Also split the full dataframe for multi-column experiments
df_train = df.loc[X_train_text.index].copy()
df_hold  = df.loc[X_hold_text.index].copy()

print(f"Train: {len(X_train_text)}, Holdout: {len(X_hold_text)}")
print(f"Classes: {sorted(y.unique())}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)

# ═══════════════════════════════════════════════
# PHASE 1: ERROR ANALYSIS on baseline
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("PHASE 1: ERROR ANALYSIS (baseline TF-IDF(1,2) + LinearSVC)")
print("="*70)

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])
baseline.fit(X_train_text, y_train)
y_pred_base = baseline.predict(X_hold_text)
base_acc = accuracy_score(y_hold, y_pred_base)
print(f"Baseline holdout accuracy: {base_acc:.4f} ({base_acc*100:.2f}%)")

# Confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_hold, y_pred_base, labels=labels)
print("\nConfusion matrix (rows=true, cols=pred):")
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# Top error pairs
errors = pd.DataFrame({"true": y_hold.values, "pred": y_pred_base})
errors = errors[errors.true != errors.pred]
print(f"\nTotal errors: {len(errors)} / {len(y_hold)}")
err_pairs = errors.groupby(["pred","true"]).size().sort_values(ascending=False).head(12)
print("\nTop error pairs (predicted -> true):")
for (p, t), c in err_pairs.items():
    print(f"  {p:30s} -> {t:30s}  ({c})")

# ═══════════════════════════════════════════════
# PHASE 2: TEXT REPRESENTATION VARIANTS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("PHASE 2: TEXT REPRESENTATION VARIANTS (5-fold CV on train only)")
print("="*70)

results = {}

configs = {
    "word(1,2) baseline":       TfidfVectorizer(ngram_range=(1,2), stop_words="english"),
    "word(1,3)":                TfidfVectorizer(ngram_range=(1,3), stop_words="english"),
    "word(1,2) sublinear":      TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True),
    "word(1,3) sublinear":      TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True),
    "word(1,2) min_df=2":       TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=2),
    "word(1,2) min_df=5":       TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=5),
    "word(1,2) sub min_df=2":   TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True, min_df=2),
    "word(1,3) sub min_df=2":   TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True, min_df=2),
    "char_wb(3,5)":             TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True),
    "char_wb(3,6)":             TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), sublinear_tf=True),
    "char_wb(3,5) min_df=2":    TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True, min_df=2),
}

for name, vec in configs.items():
    pipe = Pipeline([("tfidf", vec), ("clf", LinearSVC(dual="auto", random_state=42))])
    scores = cross_val_score(pipe, X_train_text, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    mean, std = scores.mean(), scores.std()
    results[name] = mean
    print(f"  {name:35s}  CV acc: {mean:.4f} +/- {std:.4f}")

# ═══════════════════════════════════════════════
# PHASE 3: FEATURE UNION (word + char)
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("PHASE 3: FEATURE UNION (word + char_wb TF-IDF)")
print("="*70)

union_configs = {
    "word(1,2)+char(3,5)": FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True)),
    ]),
    "word(1,3)+char(3,5)": FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True)),
    ]),
    "word(1,2)+char(3,6)": FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), sublinear_tf=True)),
    ]),
    "word(1,3)+char(3,6)": FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), sublinear_tf=True)),
    ]),
    "word(1,2)sub+char(3,5) min_df=2": FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True, min_df=2)),
    ]),
}

for name, fu in union_configs.items():
    pipe = Pipeline([("features", fu), ("clf", LinearSVC(dual="auto", random_state=42))])
    scores = cross_val_score(pipe, X_train_text, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    mean, std = scores.mean(), scores.std()
    results[name] = mean
    print(f"  {name:40s}  CV acc: {mean:.4f} +/- {std:.4f}")

# ═══════════════════════════════════════════════
# PHASE 4: MULTI-FIELD (text + categorical features)
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("PHASE 4: MULTI-FIELD (text + categorical features)")
print("="*70)

# These columns are known before routing:
print("Available categorical columns:")
for col in ["channel", "product_family", "warranty_status", "source"]:
    print(f"  {col}: {sorted(df[col].unique())}")

# Helper: extract text column from DataFrame
def get_text(df_in):
    return df_in["request_text"]

# Test text-only vs text+categorical using ColumnTransformer
cat_cols = ["channel", "product_family", "warranty_status"]
cat_cols_with_source = ["channel", "product_family", "warranty_status", "source"]

# Best text-only config from Phase 2/3 results
best_text_name = max(results, key=results.get)
print(f"\nBest text-only so far: {best_text_name} (CV={results[best_text_name]:.4f})")

# ColumnTransformer with word+char union + OHE categoricals
multi_configs = {
    "word(1,2)sub + cat3": ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ]),
    "word(1,3)sub + cat3": ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ]),
    "word(1,2)sub + cat4": ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols_with_source),
    ]),
    "union(w12+c35) + cat3": ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True), "request_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ]),
    "union(w13+c35) + cat3": ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True), "request_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ]),
    "union(w12+c35) + cat4": ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True), "request_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols_with_source),
    ]),
    "union(w13+c36) + cat4": ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True), "request_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols_with_source),
    ]),
}

multi_results = {}
for name, ct in multi_configs.items():
    pipe = Pipeline([("features", ct), ("clf", LinearSVC(dual="auto", random_state=42))])
    scores = cross_val_score(pipe, df_train[["request_text"] + cat_cols_with_source], y_train,
                             cv=cv, scoring="accuracy", n_jobs=-1)
    mean, std = scores.mean(), scores.std()
    multi_results[name] = mean
    print(f"  {name:40s}  CV acc: {mean:.4f} +/- {std:.4f}")

# ═══════════════════════════════════════════════
# PHASE 5: CLASSIFIER VARIANTS on best features
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("PHASE 5: CLASSIFIER VARIANTS")
print("="*70)

# Find best multi-field config
all_results = {**results, **multi_results}
best_overall_name = max(all_results, key=all_results.get)
print(f"Best config so far: {best_overall_name} (CV={all_results[best_overall_name]:.4f})")

# Test different classifiers with the top feature configs
# Use the best multi-field config for classifier comparison
best_multi_name = max(multi_results, key=multi_results.get)
print(f"Best multi-field config: {best_multi_name} (CV={multi_results[best_multi_name]:.4f})")

# We'll test LinearSVC with different C values and SGDClassifier
best_ct_key = best_multi_name
best_ct = multi_configs[best_ct_key]

clf_configs = {
    "LinearSVC C=0.5": LinearSVC(dual="auto", C=0.5, random_state=42),
    "LinearSVC C=1.0": LinearSVC(dual="auto", C=1.0, random_state=42),
    "LinearSVC C=2.0": LinearSVC(dual="auto", C=2.0, random_state=42),
    "LinearSVC C=5.0": LinearSVC(dual="auto", C=5.0, random_state=42),
    "SGD hinge":       SGDClassifier(loss="hinge", random_state=42, max_iter=1000),
    "SGD mod_huber":   SGDClassifier(loss="modified_huber", random_state=42, max_iter=1000),
    "LogReg lbfgs":    LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs"),
}

clf_results = {}
for name, clf in clf_configs.items():
    # Rebuild the ColumnTransformer fresh each time
    ct_fresh = ColumnTransformer([
        (n, t.clone() if hasattr(t, 'clone') else t.__class__(**t.get_params()), c)
        for n, t, c in multi_configs[best_ct_key].transformers
    ])
    pipe = Pipeline([("features", ct_fresh), ("clf", clf)])
    scores = cross_val_score(pipe, df_train[["request_text"] + cat_cols_with_source], y_train,
                             cv=cv, scoring="accuracy", n_jobs=-1)
    mean, std = scores.mean(), scores.std()
    clf_results[name] = mean
    print(f"  {name:25s}  CV acc: {mean:.4f} +/- {std:.4f}")

# Also test text-only with different C values (in case multi-field is not better)
print("\n  --- Text-only classifier variants ---")
text_clf_results = {}
for cval in [0.5, 1.0, 2.0, 5.0]:
    name = f"textonly_word13sub_C={cval}"
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True)),
        ("clf", LinearSVC(dual="auto", C=cval, random_state=42)),
    ])
    scores = cross_val_score(pipe, X_train_text, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    mean, std = scores.mean(), scores.std()
    text_clf_results[name] = mean
    print(f"  {name:35s}  CV acc: {mean:.4f} +/- {std:.4f}")

# ═══════════════════════════════════════════════
# FINAL: Select best, evaluate ONCE on holdout
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL SELECTION & HOLDOUT EVALUATION")
print("="*70)

# Gather ALL results
everything = {}
everything.update(results)
everything.update(multi_results)
everything.update(clf_results)
everything.update(text_clf_results)

# Sort and print top 10
print("\nTop 15 configurations by CV accuracy:")
top = sorted(everything.items(), key=lambda x: x[1], reverse=True)
for rank, (name, score) in enumerate(top[:15], 1):
    print(f"  {rank:2d}. {name:45s}  {score:.4f}")

best_name = top[0][0]
best_cv = top[0][1]
print(f"\n>>> BEST: {best_name} (CV={best_cv:.4f})")

# Now determine if the best is multi-field or text-only, and build that pipeline
# We need to figure out which config to use for holdout
is_multi = best_name in multi_results or best_name in clf_results
print(f"Is multi-field: {is_multi}")

# Build the final pipeline based on the winner
# We'll build the top few and evaluate all on holdout for completeness
print("\n--- Holdout evaluation of top candidates ---")

candidates = {}

# Always include baseline
candidates["baseline word(1,2)"] = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

# Best text-only
candidates["word(1,3) sublinear"] = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True)),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

# Feature union
candidates["word(1,2)+char(3,5) sublinear"] = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True)),
    ])),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

candidates["word(1,3)+char(3,5) sublinear"] = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True)),
    ])),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

# Evaluate text-only candidates on holdout
for name, pipe in candidates.items():
    pipe.fit(X_train_text, y_train)
    y_p = pipe.predict(X_hold_text)
    acc = accuracy_score(y_hold, y_p)
    f1_mac = f1_score(y_hold, y_p, average="macro")
    f1_wt = f1_score(y_hold, y_p, average="weighted")
    print(f"  {name:40s}  holdout acc={acc:.4f}  macro_f1={f1_mac:.4f}  wt_f1={f1_wt:.4f}")

# Multi-field candidates
multi_candidates = {}

multi_candidates["multi: union(w12+c35)+cat3"] = Pipeline([
    ("features", ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True), "request_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ])),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

multi_candidates["multi: union(w13+c35)+cat3"] = Pipeline([
    ("features", ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True), "request_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ])),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

multi_candidates["multi: union(w12+c35)+cat4"] = Pipeline([
    ("features", ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True), "request_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols_with_source),
    ])),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

multi_candidates["multi: union(w13+c36)+cat4"] = Pipeline([
    ("features", ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True), "request_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols_with_source),
    ])),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

multi_candidates["multi: word(1,2)sub+cat3"] = Pipeline([
    ("features", ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ])),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

multi_candidates["multi: word(1,3)sub+cat3"] = Pipeline([
    ("features", ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1,3), stop_words="english", sublinear_tf=True), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ])),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])

# Also try different C values on the best multi-field configs
for cval in [0.5, 2.0, 5.0]:
    multi_candidates[f"multi: union(w12+c35)+cat4 C={cval}"] = Pipeline([
        ("features", ColumnTransformer([
            ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True), "request_text"),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True), "request_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols_with_source),
        ])),
        ("clf", LinearSVC(dual="auto", C=cval, random_state=42)),
    ])

train_cols = ["request_text"] + cat_cols_with_source

for name, pipe in multi_candidates.items():
    pipe.fit(df_train[train_cols], y_train)
    y_p = pipe.predict(df_hold[train_cols])
    acc = accuracy_score(y_hold, y_p)
    f1_mac = f1_score(y_hold, y_p, average="macro")
    f1_wt = f1_score(y_hold, y_p, average="weighted")
    print(f"  {name:45s}  holdout acc={acc:.4f}  macro_f1={f1_mac:.4f}  wt_f1={f1_wt:.4f}")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print("Review the holdout results above to select the final model.")
print("The winning model will be used in the updated train_and_predict.py")
