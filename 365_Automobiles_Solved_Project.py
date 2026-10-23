# -*- coding: utf-8 -*-
"""
365 Automobiles — Supervised Machine Learning Streamlit App

Expected local files (place beside this script):
    365_Automobiles_Sales.csv
    365_Automobiles_Service.csv

Run:
    python -m pip install -r requirements.txt
    python -m streamlit run app.py
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# ---------------------------------------------------------------------
# 1. App configuration and data loading
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="365 Automobiles Analytics",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
SALES_FILE = BASE_DIR / "365_Automobiles_Sales.csv"
SERVICE_FILE = BASE_DIR / "365_Automobiles_Service.csv"


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes."""
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_local_csv(path_string: str) -> pd.DataFrame:
    """Read a local CSV file."""
    return pd.read_csv(path_string)


def load_dataset(uploaded_file, local_path: Path):
    """Prefer uploaded file; otherwise read the expected local CSV."""
    if uploaded_file is not None:
        return read_csv_bytes(uploaded_file.getvalue())
    if local_path.exists():
        return read_local_csv(str(local_path))
    return None


def show_df(df: pd.DataFrame, height: int = 360):
    """Consistent Streamlit dataframe display."""
    st.dataframe(df, use_container_width=True, height=height)


def show_fig(fig):
    """Render and close a Matplotlib figure in Streamlit."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def clean_yes_no(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
    )


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if "date" in col.lower() and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build numeric/categorical preprocessing based on training features."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))
    if not transformers:
        raise ValueError("No usable numeric or categorical features were found.")
    return ColumnTransformer(transformers, remainder="drop")


def engineer_dates(X: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to numeric calendar features."""
    X = X.copy()
    for col in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_dayofweek"] = X[col].dt.dayofweek
            X.drop(columns=col, inplace=True)
    return X


def drop_high_cardinality(X: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
    """Drop object/category columns with many unique values (likely IDs/free text)."""
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    high_card = [c for c in cat_cols if X[c].nunique(dropna=True) > threshold]
    return X.drop(columns=high_card, errors="ignore")


def data_summary(df: pd.DataFrame, name: str):
    st.subheader(f"{name} — data summary")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    d.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    with st.expander(f"{name}: missing values and data types", expanded=False):
        missing = df.isna().sum().sort_values(ascending=False).head(30)
        show_df(missing.to_frame("missing_count"))
        show_df(df.dtypes.astype(str).to_frame("dtype"))


def plot_bar(series: pd.Series, title: str, x_label: str = "", y_label: str = "Count"):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    show_fig(fig)


def feature_importance_table(pipeline, top_n: int = 20) -> pd.DataFrame:
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_
        return (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


# ---------------------------------------------------------------------
# 2. Sidebar data upload and navigation
# ---------------------------------------------------------------------
st.title("🚘 365 Automobiles — Analytics & Machine Learning")
st.caption("Sales conversion, vehicle pricing, vehicle-model prediction, and after-sales service analysis.")

with st.sidebar:
    st.header("Data files")
    st.caption("Upload CSVs here, or place the files beside app.py.")
    uploaded_sales = st.file_uploader(
        "Upload 365_Automobiles_Sales.csv", type=["csv"], key="sales_upload"
    )
    uploaded_service = st.file_uploader(
        "Upload 365_Automobiles_Service.csv", type=["csv"], key="service_upload"
    )

    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Data Quality",
            "Sales Analysis",
            "Price Regression",
            "Lead Conversion",
            "Vehicle Model Prediction",
            "Service Analysis",
            "Sales-Service Join",
            "Recommendations",
        ],
    )

try:
    sales = load_dataset(uploaded_sales, SALES_FILE)
    service = load_dataset(uploaded_service, SERVICE_FILE)
except Exception as exc:
    st.error(f"Could not read a CSV file: {exc}")
    st.stop()

if sales is None or service is None:
    st.warning("Both sales and service datasets are required to run this dashboard.")
    missing_files = []
    if sales is None:
        missing_files.append("`365_Automobiles_Sales.csv`")
    if service is None:
        missing_files.append("`365_Automobiles_Service.csv`")
    st.markdown("Missing: " + ", ".join(missing_files))
    st.info(
        "Upload both CSV files using the sidebar, or copy them into the same folder as `app.py`."
    )
    st.stop()

sales = parse_dates(sales)
service = parse_dates(service)

# Validate columns needed for core models
if "purchased" not in sales.columns:
    st.error("The Sales CSV must contain a `purchased` column with Yes/No values.")
    st.stop()

# ---------------------------------------------------------------------
# 3. Overview
# ---------------------------------------------------------------------
if page == "Overview":
    st.header("Dashboard Overview")
    st.write(
        "This app analyzes sales enquiries and service records. Model results are "
        "calculated from the uploaded/local data when you open the relevant page."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales records", f"{len(sales):,}")
    c2.metric("Service records", f"{len(service):,}")
    c3.metric("Sales columns", f"{sales.shape[1]:,}")
    c4.metric("Service columns", f"{service.shape[1]:,}")

    converted = clean_yes_no(sales["purchased"])
    known = converted.notna()
    conversion_rate = converted[known].mean() * 100 if known.any() else 0
    buyers_count = int((converted == 1).sum())
    nonbuyers_count = int((converted == 0).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Converted buyers", f"{buyers_count:,}")
    m2.metric("Non-buyers", f"{nonbuyers_count:,}")
    m3.metric("Conversion rate", f"{conversion_rate:.1f}%")

    st.subheader("Sales data preview")
    show_df(sales.head(10))
    st.subheader("Service data preview")
    show_df(service.head(10))

    st.download_button(
        "Download sales data as CSV",
        data=sales.to_csv(index=False).encode("utf-8"),
        file_name="sales_data_preview.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download service data as CSV",
        data=service.to_csv(index=False).encode("utf-8"),
        file_name="service_data_preview.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# 4. Data quality
# ---------------------------------------------------------------------
elif page == "Data Quality":
    st.header("Data Quality and Understanding")
    data_summary(sales, "Sales")
    data_summary(service, "Service")

    st.subheader("Date ranges")
    date_rows = []
    for name, frame in [("Sales", sales), ("Service", service)]:
        for col in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[col]):
                date_rows.append({
                    "dataset": name,
                    "date_column": col,
                    "minimum": frame[col].min(),
                    "maximum": frame[col].max(),
                    "missing_dates": int(frame[col].isna().sum()),
                })
    if date_rows:
        show_df(pd.DataFrame(date_rows))
    else:
        st.info("No date columns were detected.")

    st.subheader("Outcome populations")
    for col in ["purchased", "model_purchased"]:
        if col in sales.columns:
            st.write(f"Counts for `{col}`")
            show_df(sales[col].value_counts(dropna=False).rename("count").to_frame())

    if "final_price_aed" in sales.columns:
        st.write("Missing final price by purchase status")
        show_df(
            sales.groupby("purchased", dropna=False)["final_price_aed"]
            .apply(lambda values: values.isna().sum())
            .to_frame("missing_final_price")
        )

# ---------------------------------------------------------------------
# 5. Sales analysis
# ---------------------------------------------------------------------
elif page == "Sales Analysis":
    st.header("Sales and Enquiry Analysis")

    for col in ["lead_source", "showroom", "model_interested", "model_purchased"]:
        if col in sales.columns:
            st.subheader(f"Enquiries by {col}")
            counts = sales[col].fillna("Missing").value_counts(dropna=False).head(20)
            show_df(counts.rename("enquiries").to_frame())
            plot_bar(counts.sort_values(ascending=False), f"Enquiries by {col}", col, "Enquiries")

    converted_sales = sales.copy()
    converted_sales["_converted"] = clean_yes_no(converted_sales["purchased"])
    for col in ["lead_source", "showroom", "model_interested"]:
        if col in converted_sales.columns:
            summary = (
                converted_sales.dropna(subset=["_converted"])
                .groupby(col)["_converted"]
                .agg(enquiries="count", conversion_rate="mean")
                .sort_values("conversion_rate", ascending=False)
            )
            if not summary.empty:
                summary["conversion_rate_pct"] = summary["conversion_rate"] * 100
                st.subheader(f"Conversion rate by {col}")
                show_df(summary.round(3))
                plot_bar(
                    summary["conversion_rate_pct"],
                    f"Conversion rate by {col}",
                    col,
                    "Conversion rate (%)",
                )

    if "model_purchased" in sales.columns:
        buyer_mask = clean_yes_no(sales["purchased"]) == 1
        mix = sales.loc[buyer_mask, "model_purchased"].value_counts(normalize=True).mul(100).round(2)
        if not mix.empty:
            st.subheader("Purchased vehicle model mix")
            show_df(mix.rename("share_pct").to_frame())
            plot_bar(mix, "Purchased vehicle model mix", "Vehicle model", "Share (%)")

    date_candidates = [c for c in sales.columns if "date" in c.lower()]
    if date_candidates:
        date_col = date_candidates[0]
        dates = pd.to_datetime(sales[date_col], errors="coerce")
        valid_dates = dates.notna()
        if valid_dates.any():
            monthly_frame = pd.DataFrame({
                "month": dates[valid_dates].dt.to_period("M").astype(str),
                "converted": clean_yes_no(sales.loc[valid_dates, "purchased"]).values,
            })
            monthly = monthly_frame.groupby("month").agg(
                enquiries=("month", "size"),
                conversions=("converted", "sum"),
            )
            monthly["conversion_rate_pct"] = (
                monthly["conversions"] / monthly["enquiries"] * 100
            )
            st.subheader("Monthly enquiry and conversion trend")
            show_df(monthly.reset_index())
            fig, ax = plt.subplots(figsize=(10, 4.5))
            monthly[["enquiries", "conversions"]].plot(ax=ax, marker="o")
            ax.set_title("Monthly enquiries vs conversions")
            ax.set_xlabel("Month")
            ax.set_ylabel("Records")
            fig.tight_layout()
            show_fig(fig)

# ---------------------------------------------------------------------
# 6. Linear regression: final selling price
# ---------------------------------------------------------------------
elif page == "Price Regression":
    st.header("Model A — Final Selling Price (Linear Regression)")
    st.caption("Trains on converted buyers only. Target: final_price_aed.")

    required = {"purchased", "final_price_aed"}
    if not required.issubset(sales.columns):
        st.error("Sales CSV needs `purchased` and `final_price_aed` columns.")
        st.stop()

    buyers = sales[clean_yes_no(sales["purchased"]) == 1].copy()
    buyers["final_price_aed"] = pd.to_numeric(buyers["final_price_aed"], errors="coerce")
    buyers = buyers.dropna(subset=["final_price_aed"])

    excluded = {
        "final_price_aed", "vehicle_id", "days_to_delivery", "discount_pct",
        "down_payment_aed", "model_purchased", "purchased",
    }
    feature_cols = [c for c in buyers.columns if c not in excluded]
    X = engineer_dates(buyers[feature_cols])
    X = drop_high_cardinality(X)
    y = buyers["final_price_aed"]

    if len(buyers) < 5 or X.shape[1] == 0:
        st.warning("Not enough buyer records or usable features to train the regression model.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    preprocessor = make_preprocessor(X_train)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    with st.spinner("Training Linear Regression model..."):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    c1, c2, c3 = st.columns(3)
    c1.metric("R-squared", f"{r2:.3f}")
    c2.metric("MAE (AED)", f"{mae:,.2f}")
    c3.metric("RMSE (AED)", f"{rmse:,.2f}")

    metrics = pd.DataFrame({
        "Metric": ["R-squared", "MAE (AED)", "RMSE (AED)"],
        "Value": [r2, mae, rmse],
    })
    st.subheader("Regression metrics")
    show_df(metrics)

    actual_pred = pd.DataFrame({
        "Actual price (AED)": y_test.to_numpy(),
        "Predicted price (AED)": predictions,
        "Residual (actual - predicted)": y_test.to_numpy() - predictions,
    })
    st.subheader("Actual vs predicted prices")
    show_df(actual_pred.head(100))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, predictions, alpha=0.5)
    low = min(float(y_test.min()), float(np.min(predictions)))
    high = max(float(y_test.max()), float(np.max(predictions)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual final price (AED)")
    ax.set_ylabel("Predicted final price (AED)")
    ax.set_title("Linear Regression — Actual vs Predicted")
    fig.tight_layout()
    show_fig(fig)

    st.download_button(
        "Download actual vs predicted prices",
        data=actual_pred.to_csv(index=False).encode("utf-8"),
        file_name="price_regression_predictions.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# 7. Binary classification: lead conversion
# ---------------------------------------------------------------------
elif page == "Lead Conversion":
    st.header("Model B — Lead Conversion (Random Forest)")
    st.caption("Population: all enquiries. Target: purchased (Yes/No).")

    exclude = {
        "purchased", "model_purchased", "final_price_aed", "days_to_delivery",
        "vehicle_id", "financing_approved", "discount_pct", "payment_type",
        "down_payment_pct", "down_payment_aed",
    }
    y = clean_yes_no(sales["purchased"])
    valid = y.notna()
    X = sales.loc[valid, [c for c in sales.columns if c not in exclude]].copy()
    y = y.loc[valid].astype(int)
    X = engineer_dates(X)
    X = drop_high_cardinality(X)

    if y.nunique() < 2 or len(y) < 10 or X.shape[1] == 0:
        st.warning("The dataset needs at least 10 labelled records, both Yes/No outcomes, and usable features.")
        st.stop()

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=stratify
    )
    model = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    with st.spinner("Training Random Forest lead conversion model..."):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, list(model.named_steps["model"].classes_).index(1)]

    baseline_class = int(y_train.mode().iloc[0])
    baseline_predictions = np.full(len(y_test), baseline_class)
    try:
        auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    except Exception:
        auc = np.nan

    metrics = pd.DataFrame({
        "Metric": [
            "Majority baseline accuracy", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
        ],
        "Value": [
            accuracy_score(y_test, baseline_predictions),
            accuracy_score(y_test, predictions),
            precision_score(y_test, predictions, zero_division=0),
            recall_score(y_test, predictions, zero_division=0),
            f1_score(y_test, predictions, zero_division=0),
            auc,
        ],
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}")
    c2.metric("F1 score", f"{f1_score(y_test, predictions, zero_division=0):.3f}")
    c3.metric("ROC-AUC", "N/A" if pd.isna(auc) else f"{auc:.3f}")

    st.subheader("Model metrics and baseline")
    show_df(metrics)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(ax=ax, colorbar=False)
    ax.set_title("Random Forest — Lead Conversion")
    fig.tight_layout()
    show_fig(fig)

    st.subheader("Classification report")
    report = classification_report(
        y_test, predictions, labels=[0, 1], target_names=["No", "Yes"],
        zero_division=0, output_dict=True,
    )
    show_df(pd.DataFrame(report).transpose().round(3))

    importance = feature_importance_table(model)
    if not importance.empty:
        st.subheader("Top feature importances")
        show_df(importance)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(importance["feature"][::-1], importance["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Lead Conversion — Random Forest Feature Importance")
        fig.tight_layout()
        show_fig(fig)

# ---------------------------------------------------------------------
# 8. Multiclass classification: vehicle model purchased
# ---------------------------------------------------------------------
elif page == "Vehicle Model Prediction":
    st.header("Model C — Vehicle Model Purchased (Random Forest)")
    st.caption("Population: converted buyers only. Target: model_purchased.")

    if "model_purchased" not in sales.columns:
        st.error("Sales CSV needs a `model_purchased` column.")
        st.stop()

    buyers = sales[clean_yes_no(sales["purchased"]) == 1].copy()
    buyers = buyers.dropna(subset=["model_purchased"])
    exclude = {
        "model_purchased", "purchased", "final_price_aed", "vehicle_id",
        "days_to_delivery", "financing_approved", "discount_pct",
        "payment_type", "down_payment_pct", "down_payment_aed",
    }
    X = buyers[[c for c in buyers.columns if c not in exclude]].copy()
    y = buyers["model_purchased"].astype(str)
    X = drop_high_cardinality(engineer_dates(X))

    if len(y) < 10 or y.nunique() < 2 or X.shape[1] == 0:
        st.warning("Not enough converted buyers or target classes to train this model.")
        st.stop()

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=stratify
    )
    model = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    with st.spinner("Training Random Forest vehicle model predictor..."):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Weighted precision", f"{precision:.3f}")
    c3.metric("Weighted recall", f"{recall:.3f}")
    c4.metric("Weighted F1", f"{f1:.3f}")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"],
        "Value": [acc, precision, recall, f1],
    })
    st.subheader("Model metrics")
    show_df(metrics)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title("Random Forest — Vehicle Model Purchased")
    fig.tight_layout()
    show_fig(fig)

    st.subheader("Classification report")
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    show_df(pd.DataFrame(report).transpose().round(3))

    actual_pred = pd.DataFrame({"Actual model": y_test.to_numpy(), "Predicted model": predictions})
    st.subheader("Test-set predictions")
    show_df(actual_pred.head(100))

    mismatches = actual_pred[actual_pred["Actual model"] != actual_pred["Predicted model"]]
    if not mismatches.empty:
        st.subheader("Examples of incorrect predictions")
        show_df(mismatches.head(30))

    importance = feature_importance_table(model)
    if not importance.empty:
        st.subheader("Top feature importances")
        show_df(importance)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(importance["feature"][::-1], importance["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Vehicle Model — Random Forest Feature Importance")
        fig.tight_layout()
        show_fig(fig)

    st.download_button(
        "Download test-set predictions",
        data=actual_pred.to_csv(index=False).encode("utf-8"),
        file_name="vehicle_model_predictions.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# 9. Service analysis
# ---------------------------------------------------------------------
elif page == "Service Analysis":
    st.header("After-Sales Service Analysis")

    for dim in ["model", "trim", "service_type", "service_centre", "service_center"]:
        if dim in service.columns and "service_cost_aed" in service.columns:
            cost = pd.to_numeric(service["service_cost_aed"], errors="coerce")
            summary = service.assign(_cost=cost).groupby(dim)["_cost"].agg(
                visits="count", mean_cost="mean", median_cost="median"
            ).sort_values("mean_cost", ascending=False).head(20)
            st.subheader(f"Service cost by {dim}")
            show_df(summary.round(2))
            plot_bar(summary["mean_cost"].sort_values(ascending=False), f"Average service cost by {dim}", dim, "Average cost (AED)")

    if "battery_health_pct" in service.columns:
        battery = pd.to_numeric(service["battery_health_pct"], errors="coerce")
        for x_col in ["vehicle_age_months", "odometer_km"]:
            if x_col in service.columns:
                x_values = pd.to_numeric(service[x_col], errors="coerce")
                valid = x_values.notna() & battery.notna()
                if valid.any():
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(x_values[valid], battery[valid], alpha=0.4)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel("Battery health (%)")
                    ax.set_title(f"Battery health vs {x_col}")
                    fig.tight_layout()
                    show_fig(fig)

    wait_col = next(
        (c for c in ["workshop_wait_time_hours", "workshop_wait_time"] if c in service.columns),
        None,
    )
    sat_col = "customer_satisfaction" if "customer_satisfaction" in service.columns else None
    centre_col = next(
        (c for c in ["service_centre", "service_center"] if c in service.columns),
        None,
    )

    if centre_col:
        agg = {}
        if wait_col:
            agg[wait_col] = ["count", "mean", "median"]
        if sat_col:
            agg[sat_col] = ["mean", "median"]
        if agg:
            st.subheader("Wait time and satisfaction by service centre")
            show_df(service.groupby(centre_col).agg(agg).round(2))

    if "repeat_visit_30_days" in service.columns:
        repeat = clean_yes_no(service["repeat_visit_30_days"])
        if centre_col:
            repeat_rate = (
                service.assign(_repeat=repeat)
                .groupby(centre_col)["_repeat"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
                .to_frame("repeat_visit_rate_pct")
            )
            st.subheader("Repeat visits by service centre")
            show_df(repeat_rate.round(2))
            if not repeat_rate.empty:
                plot_bar(repeat_rate["repeat_visit_rate_pct"], "Repeat visit rate by service centre", centre_col, "Repeat visits (%)")

    if not any(c in service.columns for c in ["service_cost_aed", "battery_health_pct", wait_col, sat_col, "repeat_visit_30_days"]):
        st.info("No recognized service-analysis columns were found. Check the service CSV column names.")

# ---------------------------------------------------------------------
# 10. Sales-to-service join at vehicle level
# ---------------------------------------------------------------------
elif page == "Sales-Service Join":
    st.header("Sales-to-Service Join")
    st.caption("Service visits are aggregated to one row per vehicle before joining, preventing sales-row multiplication.")

    if "vehicle_id" not in sales.columns or "vehicle_id" not in service.columns:
        st.error("Both CSV files must contain a `vehicle_id` column for this analysis.")
        st.stop()

    sales_join = sales.copy()
    service_join = service.copy()
    sales_join["vehicle_id"] = sales_join["vehicle_id"].astype("string").str.strip()
    service_join["vehicle_id"] = service_join["vehicle_id"].astype("string").str.strip()
    sales_join = sales_join.dropna(subset=["vehicle_id"]).copy()
    service_join = service_join.dropna(subset=["vehicle_id"]).copy()

    agg_dict = {"service_visits": ("vehicle_id", "size")}
    if "service_cost_aed" in service_join.columns:
        service_join["service_cost_aed"] = pd.to_numeric(
            service_join["service_cost_aed"], errors="coerce"
        )
        agg_dict["total_service_cost_aed"] = ("service_cost_aed", "sum")
        agg_dict["avg_service_cost_aed"] = ("service_cost_aed", "mean")

    service_vehicle = service_join.groupby("vehicle_id", as_index=False).agg(**agg_dict)

    st.subheader("Join diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sales rows", f"{len(sales_join):,}")
    c2.metric("Service rows", f"{len(service_join):,}")
    c3.metric("Vehicles with service", f"{service_vehicle['vehicle_id'].nunique():,}")

    sales_service = sales_join.merge(
        service_vehicle, on="vehicle_id", how="left", validate="many_to_one"
    )
    st.metric("Sales records without matching service", f"{int(sales_service['service_visits'].isna().sum()):,}")

    desired = [
        "vehicle_id", "model_purchased", "service_visits",
        "total_service_cost_aed", "avg_service_cost_aed",
    ]
    display_cols = [c for c in desired if c in sales_service.columns]
    st.subheader("Joined data preview")
    show_df(sales_service[display_cols].head(100))

    st.download_button(
        "Download joined Sales-Service dataset",
        data=sales_service.to_csv(index=False).encode("utf-8"),
        file_name="365_Automobiles_Sales_Service_Joined.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# 11. Management recommendations
# ---------------------------------------------------------------------
elif page == "Recommendations":
    st.header("Management Recommendations")
    st.write("Use the model metrics and descriptive summaries above as evidence when prioritizing actions.")

    st.subheader("1. Sales prioritisation")
    st.write("Compare observed conversion rates by lead source, showroom, and model of interest. Prioritize follow-up using both conversion evidence and model results.")

    st.subheader("2. Lead scoring governance")
    st.write("Review false negatives and false positives separately. False negatives may mean missed buyers; false positives consume sales-team capacity. Do not rely on accuracy alone.")

    st.subheader("3. Pricing")
    st.write("Use the Linear Regression MAE, RMSE, and actual-versus-predicted plot to understand price-estimation error. Predictive associations are not proof of causation.")

    st.subheader("4. Stock planning")
    st.write("Review the vehicle-model confusion matrix and model mix to identify categories the model struggles to distinguish.")

    st.subheader("5. After-sales service")
    st.write("Use service cost, battery health, wait time, satisfaction, and repeat-visit summaries to investigate workshop capacity and customer experience.")

    st.warning(
        "These are decision prompts, not claims about the business results. Confirm any action against the actual data, model performance, and operational context."
    )

st.sidebar.caption("365 Automobiles Analytics • Random state = 42")
