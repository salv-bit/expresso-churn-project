# train_expresso.py
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from joblib import dump

# ========= CONFIG =========
FILE_PATH = r"C:\Users\DELL LATITUDE E7250\Downloads\Expresso_churn_dataset.csv"
TARGET_NAME = "CHURN"                 # set your target column name explicitly
TEST_SIZE   = 0.2
SEED        = 42
HIGH_CARD_THRESHOLD = 300             # drop very wide categoricals
SAMPLE_ROWS = None                    # set e.g. 60000 if RAM is tight; None = use all
# ==========================


def load_csv_safely(path, nrows=None):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows)
        except Exception:
            continue
    return pd.read_csv(path, nrows=nrows)


# 1) LOAD
df = load_csv_safely(FILE_PATH, nrows=SAMPLE_ROWS)
print("\n===== GENERAL INFO =====")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

if TARGET_NAME not in df.columns:
    raise RuntimeError(f"Target '{TARGET_NAME}' not found. Columns: {list(df.columns)}")

# Basic info/describe/missing (prints)
print("\n--- df.info() ---")
print(df.info())
print("\n--- df.describe(numeric) ---")
print(df.select_dtypes(include=[np.number]).describe().T.head(20))
print("\n--- Missing values (top 20) ---")
print(df.isna().sum().sort_values(ascending=False).head(20))

# 2) LIGHT CLEANING
# remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"\nDropped duplicates: {before - len(df)}")

# small outlier handling: winsorize numeric columns to [1%, 99%]
num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols_all:
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    df[col] = df[col].clip(q1, q99)

# drop rows with missing target
df = df.dropna(subset=[TARGET_NAME])

# 3) SPLIT X / Y
y = df[TARGET_NAME]
X = df.drop(columns=[TARGET_NAME])

# map common string targets to 0/1 if needed
if y.dtype == "O":
    mapping = {"yes":1,"no":0,"true":1,"false":0,"churned":1,"not churned":0,"1":1,"0":0}
    y = y.astype(str).str.strip().str.lower().map(mapping).fillna(y)
if not np.issubdtype(y.dtype, np.number):
    y, _ = pd.factorize(y)

# 4) DROP HEAVY CATEGORICALS (and IDs)
id_like = ["user_id","USER_ID","msisdn","MSISDN","id","ID"]
drop_cols = [c for c in id_like if c in X.columns]
obj = X.select_dtypes(include="object")
if not obj.empty:
    card = obj.nunique().sort_values(ascending=False)
    high_card_cols = card[card > HIGH_CARD_THRESHOLD].index.tolist()
    drop_cols += high_card_cols
if drop_cols:
    print("Dropping columns:", drop_cols)
    X = X.drop(columns=drop_cols, errors="ignore")

# Identify columns by type after drops
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

# 5) PREPROCESSOR + MODEL
numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler(with_mean=False))  # sparse-friendly
])
categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols)
    ],
    remainder="drop"
)

# Classifier (sparse-friendly logistic regression)
clf = LogisticRegression(max_iter=300, solver="saga", n_jobs=-1, random_state=SEED)

pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

# 6) TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

print("\nTraining...")
pipe.fit(X_train, y_train)

# 7) EVALUATE
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
y_prob = None
auc = None
try:
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
except Exception:
    pass

print("\n===== EVALUATION =====")
print("Accuracy:", round(acc, 4))
if auc is not None:
    print("ROC-AUC:", round(auc, 4))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 9) SAVE ARTIFACTS
os.makedirs("artifacts", exist_ok=True)
model_path = os.path.join("artifacts", "churn_model.joblib")
dump(pipe, model_path)
print("Saved model to:", model_path)

# Save metrics
metrics = {"accuracy": float(acc), "roc_auc": float(auc) if auc is not None else None}
with open(os.path.join("artifacts", "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save feature metadata for Streamlit (types + categorical choices)
cat_choices = {}
for col in cat_cols:
    # store up to 50 most common categories to keep UI tidy
    top = X[col].value_counts(dropna=True).head(50).index.tolist()
    cat_choices[col] = top

meta = {
    "numeric_columns": num_cols,
    "categorical_columns": cat_cols,
    "categorical_choices": cat_choices,
    "dropped_columns": drop_cols,
    "target": TARGET_NAME
}
with open(os.path.join("artifacts", "feature_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Saved artifacts/feature_metadata.json")

print("\n✅ Training complete.")

#2) app.py (Streamlit UI)

#This reads the saved artifacts and builds input widgets automatically.

# app.py
import json
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from pathlib import Path

st.set_page_config(page_title="Expresso Churn Predictor", page_icon="📶", layout="centered")

MODEL_PATH = Path("artifacts/churn_model.joblib")
META_PATH  = Path("artifacts/feature_metadata.json")

st.title("📶 Expresso Churn Prediction")
st.write("Enter customer features below and click **Predict**.")

# Load model & metadata
if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Model or metadata not found. Please run `python train_expresso.py` first.")
    st.stop()

pipe = load(MODEL_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8"))

num_cols = meta.get("numeric_columns", [])
cat_cols = meta.get("categorical_columns", [])
cat_choices = meta.get("categorical_choices", {})
target = meta.get("target", "CHURN")

# Build input form
with st.form("input_form"):
    st.subheader("Input Features")

    # make two columns for nicer layout
    colA, colB = st.columns(2)

    inputs = {}

    # numeric inputs
    for i, col in enumerate(num_cols):
        with (colA if i % 2 == 0 else colB):
            inputs[col] = st.number_input(col, value=0.0)

    # categorical inputs
    for i, col in enumerate(cat_cols):
        choices = cat_choices.get(col, [])
        default = choices[0] if choices else ""
        with (colA if i % 2 == 0 else colB):
            if choices:
                inputs[col] = st.selectbox(col, choices, index=0)
            else:
                # free text if we didn't save options
                inputs[col] = st.text_input(col, value=default)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build one-row DataFrame in the same schema the model expects
    X_input = pd.DataFrame({k: [v] for k, v in inputs.items()})

    try:
        proba = None
        # prefer predict_proba if available
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_input)[:, 1][0]
            pred  = int(proba >= 0.5)
        else:
            # fallback: decision_function + sigmoid
            from scipy.special import expit
            scores = pipe.decision_function(X_input)
            score1 = scores if np.ndim(scores) == 0 else (scores[:, 1] if scores.ndim > 1 else scores)
            proba = float(expit(score1)[0])
            pred  = int(proba >= 0.5)

        st.success(f"Prediction: **{'Churn' if pred==1 else 'Not Churn'}**")
        st.metric("Churn probability", f"{proba*100:.2f}%")
    except Exception as e:
        st.error(f"Failed to predict: {e}")


import json, os

# ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# collect metadata from training variables
meta = {
    "numeric_columns": num_cols,
    "categorical_columns": cat_cols,
    "categorical_choices": {},  # optional, leave empty for now
    "target": "CHURN"
}

with open("artifacts/columns_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("✅ Saved columns_meta.json")
