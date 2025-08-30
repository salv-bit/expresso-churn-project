import os
import json
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

# Profiling
from ydata_profiling import ProfileReport

# ML
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Optional: download from Google Drive by file ID
import gdown

# -----------------------
# CONFIG
# -----------------------
# Your shared Drive file id:
# https://drive.google.com/file/d/14z5kEPm_SNl-Dt5qtBTgH0NWDb8iU6fz/view?usp=sharing
DRIVE_FILE_ID = os.environ.get("DRIVE_FILE_ID", "14z5kEPm_SNl-Dt5qtBTgH0NWDb8iU6fz")

RAW_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
RAW_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# If you already downloaded the dataset, set DATA_PATH to a local file (.csv/.xlsx/.zip)
DATA_PATH = os.environ.get("DATA_PATH", str(RAW_DIR / "expresso_download"))

# -----------------------
# HELPERS
# -----------------------
def download_from_drive(file_id: str, dest: str):
    """Download a file by Google Drive file ID."""
    print(f"Downloading Google Drive file id {file_id} -> {dest}")
    gdown.download(id=file_id, output=dest, quiet=False)
    return dest

def load_dataframe(path: str) -> pd.DataFrame:
    """Load CSV/XLSX, or first CSV/XLSX inside a ZIP, into a DataFrame."""
    lower = path.lower()
    if os.path.isfile(path) and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            members = z.namelist()
            csv_like = [m for m in members if m.lower().endswith((".csv", ".tsv", ".txt", ".xlsx"))]
            if not csv_like:
                raise ValueError("No CSV/XLSX found inside the ZIP.")
            member = csv_like[0]
            extract_to = RAW_DIR / Path(member).name
            z.extract(member, RAW_DIR)
            path = str(extract_to)
            lower = path.lower()

    if lower.endswith(".csv"):
        df = pd.read_csv(path)
    elif lower.endswith(".xlsx"):
        df = pd.read_excel(path)
    elif lower.endswith(".tsv") or lower.endswith(".txt"):
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"Unsupported file format for: {path}. Error: {e}")
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df

def guess_target_column(df: pd.DataFrame) -> str:
    # Try common names first
    candidates = ["churn", "CHURN", "Churn", "target", "Target", "TARGET", "label", "Label"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: last binary-like column
    for c in reversed(df.columns):
        nunq = df[c].nunique(dropna=True)
        if nunq == 2:
            return c
    raise ValueError("Couldn't determine the target column. Rename your target to 'churn' or set TARGET_COLUMN env var.")

def coerce_target(y: pd.Series) -> pd.Series:
    """Map typical churn/active values to 1/0; if two unique labels, map them to 0/1."""
    y2 = y.copy()
    mapping_true = {"1", "true", "yes", "y", "churned", "churn", "t"}
    mapping_false = {"0", "false", "no", "n", "active", "f", "not_churn"}
    if y2.dtype == object:
        y2 = y2.astype(str).str.strip().str.lower()
        y2 = y2.map(lambda v: 1 if v in mapping_true else (0 if v in mapping_false else v))
    if y2.dtype == object:
        uniq = sorted([u for u in y2.dropna().unique()])
        if len(uniq) == 2:
            lab_map = {uniq[0]: 0, uniq[1]: 1}
            y2 = y2.map(lab_map)
    return pd.to_numeric(y2, errors="coerce")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    # 1) Get data locally or via Drive ID
    if not os.path.exists(DATA_PATH):
        DATA_PATH = download_from_drive(DRIVE_FILE_ID, DATA_PATH)
    else:
        print(f"Using existing file at: {DATA_PATH}")

    # 2) Load into DataFrame
    df = load_dataframe(DATA_PATH)
    df = standardize_columns(df)

    # 3) Basic EDA artifacts
    overview_path = ARTIFACTS_DIR / "data_overview.txt"
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("HEAD (first 5 rows)\n")
        f.write(df.head().to_string())
        f.write("\n\nINFO\n")
        df.info(buf=f)
        f.write("\n\nDESCRIBE (numeric)\n")
        f.write(df.describe(include=[np.number]).to_string())
        f.write("\n\nMISSING VALUES\n")
        f.write(df.isna().sum().sort_values(ascending=False).to_string())
    print(f"Saved dataset overview to: {overview_path}")

    # 4) Pandas profiling (sample for speed/large data)
    sample = df.sample(n=min(len(df), 100_000), random_state=42)
    try:
        profile = ProfileReport(sample, title="Expresso Churn Profiling (sampled)", minimal=True)
        profile_out = REPORTS_DIR / "profile.html"
        profile.to_file(profile_out)
        print(f"Saved profiling report to: {profile_out}")
    except Exception as e:
        print("Profiling skipped:", e)

    # 5) Deduplicate
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")

    # 6) Target + features
    target_col = os.environ.get("TARGET_COLUMN", None)
    if not target_col or target_col not in df.columns:
        target_col = guess_target_column(df)

    y = coerce_target(df[target_col])
    X = df.drop(columns=[target_col])

    # 7) Identify feature types
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # 8) Drop ID-like columns (very high cardinality)
    id_like = [c for c in X.columns if X[c].nunique(dropna=True) / max(len(X), 1) > 0.9]
    if id_like:
        print("Dropping likely ID columns:", id_like)
        X = X.drop(columns=id_like)
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]

    # 9) Outlier handling via IQR clipping (save bounds for app sliders)
    bounds = {}
    for c in num_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lo = float(q1 - 1.5 * iqr)
        hi = float(q3 + 1.5 * iqr)
        bounds[c] = {"lo": lo, "hi": hi}
        X[c] = s.clip(lo, hi)

    # 10) Category suggestions (top-50) for Streamlit dropdowns
    cat_values = {}
    for c in cat_cols:
        vals = (
            X[c]
            .astype(str)
            .replace({"nan": None})
            .dropna()
            .value_counts()
            .head(50)
            .index
            .tolist()
        )
        cat_values[c] = vals if vals else []

    # 11) Save metadata for app
    meta = {
        "target": target_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "id_like": id_like,
        "iqr_bounds": bounds,
        "cat_values": cat_values,
    }
    with open(ARTIFACTS_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 12) Clean target & split
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 13) Preprocess + model
    numeric_pre = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pre = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pre, num_cols),
            ("cat", categorical_pre, cat_cols),
        ]
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": target_col,
    }
    print("Metrics:", metrics)

    # 14) Save artifacts
    joblib.dump(model, ARTIFACTS_DIR / "churn_model.joblib")
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Optional snapshot of cleaned data (first 10k rows)
    cleaned_path = ARTIFACTS_DIR / "cleaned_sample.csv"
    X_copy = X.copy()
    X_copy[target_col] = y
    X_copy.head(10_000).to_csv(cleaned_path, index=False)

    print(f"Artifacts saved in: {ARTIFACTS_DIR.resolve()}")
