from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_CANDIDATES = [
    "marketing_data.csv",
    "marketing-data-csv",
    "marketing_data_csv",
]


def resolve_data_path() -> Path:
    candidates = [Path(__file__).resolve().parent / name for name in DATA_CANDIDATES]
    candidates.extend(Path.cwd() / name for name in DATA_CANDIDATES)

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the marketing dataset. Expected one of: "
        + ", ".join(DATA_CANDIDATES)
    )


def clean_marketing_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
        df["Enrollment_Month"] = df["Dt_Customer"].dt.month
        df["Enrollment_Year"] = df["Dt_Customer"].dt.year
        df["Enrollmnt_Day"] = df["Dt_Customer"].dt.day

    if "Income" in df.columns:
        df["Income"] = pd.to_numeric(df["Income"], errors="coerce").fillna(0)

    if "Year_Birth" in df.columns:
        df["Year_Birth"] = pd.to_numeric(df["Year_Birth"], errors="coerce")
        df.loc[df["Year_Birth"] == 1893, "Year_Birth"] = 1993
        df.loc[df["Year_Birth"] == 1899, "Year_Birth"] = 1999
        df = df[df["Year_Birth"] != 1900].copy()

    if "Enrollment_Year" in df.columns and "Year_Birth" in df.columns:
        df["Age"] = df["Enrollment_Year"] - df["Year_Birth"]

    product_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]

    for col in product_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if all(col in df.columns for col in product_columns):
        df["Total_Purchases"] = df[product_columns].sum(axis=1)

    if "Income" in df.columns:
        df = df[df["Income"] <= 105471].copy()

    return df


def apply_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #0f172a;
                --panel: #111827;
                --muted: #94a3b8;
                --accent: #8b5cf6;
                --accent-2: #22c55e;
                --accent-3: #f59e0b;
                --card: rgba(15, 23, 42, 0.85);
                --border: rgba(148, 163, 184, 0.22);
            }

            .stApp {
                background: linear-gradient(135deg, #0f172a 0%, #111827 28%, #1e293b 100%);
                color: #e2e8f0;
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            .stMetric {
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid var(--border);
                border-radius: 0.85rem;
                padding: 0.8rem 1rem;
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.28);
            }

            div[data-testid="stMetricValue"] {
                font-size: 1.5rem;
                font-weight: 700;
                color: #f8fafc;
            }

            h1, h2, h3 {
                color: #f8fafc;
            }

            .section-label {
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #a5b4fc;
                margin-bottom: 0.3rem;
            }

            .insight-card {
                background: rgba(17, 24, 39, 0.9);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 1rem;
                padding: 1rem 1.1rem;
                margin-bottom: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_age_distribution(df: pd.DataFrame) -> go.Figure:
    if "Age" not in df.columns:
        empty = go.Figure()
        empty.update_layout(
            title="Age Distribution",
            xaxis_title="Age Group",
            yaxis_title="Number of Individuals",
            template="plotly_dark",
        )
        return empty

    min_age = float(df["Age"].min())
    max_age = float(df["Age"].max())
    age_bins = [min_age, 30, 40, 50, 60, max_age]
    age_labels = ["<30", "30-39", "40-49", "50-59", "60+"]
    df_plot = df.copy()
    df_plot["Age_group"] = pd.cut(df_plot["Age"], bins=age_bins, labels=age_labels, right=False)
    counts = df_plot["Age_group"].value_counts().sort_index()

    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="#f472b6"))
    fig.update_layout(
        title="Distribution of Age Groups",
        xaxis_title="Age Group",
        yaxis_title="Number of Individuals",
        xaxis=dict(tickangle=0),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


def build_campaign_chart(df: pd.DataFrame) -> go.Figure:
    accepted = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
    ]
    campaign_counts = df[accepted].sum()
    fig = go.Figure(
        go.Bar(
            x=campaign_counts.index,
            y=campaign_counts.values,
            marker=dict(
                color=["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"]
            ),
        )
    )
    fig.update_layout(
        title="Campaign Offer Acceptance",
        xaxis_title="Campaign",
        yaxis_title="Responses",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def build_spending_by_education(df: pd.DataFrame):
    product_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    if "Education" not in df.columns:
        return px.bar()

    education_spending = df.groupby("Education")[product_columns].sum().reset_index()
    melted = education_spending.melt(
        id_vars="Education", var_name="Product", value_name="Amount"
    )

    fig = px.bar(
        melted,
        x="Education",
        y="Amount",
        color="Product",
        barmode="group",
        template="plotly_dark",
        color_discrete_sequence=["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#c084fc", "#f87171"],
        labels={
            "Education": "Education Level",
            "Amount": "Total Spend",
            "Product": "Product Category",
        },
        title="Spending by Education Level and Product Category",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
    )
    return fig


def build_country_distribution(df: pd.DataFrame) -> go.Figure:
    if "Country" not in df.columns:
        empty = go.Figure()
        empty.update_layout(
            title="Top Customer Countries",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return empty

    country_counts = df["Country"].value_counts().head(8).sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=country_counts.values, y=country_counts.index, orientation="h", marker_color="#a78bfa"))
    fig.update_layout(
        title="Top Customer Countries",
        xaxis_title="Customers",
        yaxis_title="Country",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


def build_regression_summary(df: pd.DataFrame) -> tuple[go.Figure, dict]:
    features = [
        "Year_Birth",
        "Income",
        "Kidhome",
        "Teenhome",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "Response",
        "Complain",
        "Enrollment_Month",
        "Enrollment_Year",
        "Enrollmnt_Day",
        "Age",
        "Total_Purchases",
    ]

    model_df = df[features + ["Recency"]].copy()
    model_df = model_df.dropna()

    X = model_df[features]
    y = model_df["Recency"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            name="Actual vs. Predicted",
            marker=dict(color="#c084fc", size=10, opacity=0.8),
        )
    )
    max_val = max(float(y_test.max()), float(y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Perfect Predictions",
            line=dict(color="#fbbf24", dash="dash"),
        )
    )
    fig.update_layout(
        title="Recency Prediction Performance",
        xaxis_title="Actual Recency",
        yaxis_title="Predicted Recency",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )

    return fig, {"mse": mse, "r2": r2}


def get_overview_metrics(df: pd.DataFrame) -> dict:
    metrics = {
        "customers": len(df),
        "avg_income": float(df["Income"].mean()) if "Income" in df.columns else 0.0,
        "avg_recency": float(df["Recency"].mean()) if "Recency" in df.columns else 0.0,
        "campaign_response_rate": (
            float(df["Response"].mean() * 100) if "Response" in df.columns else 0.0
        ),
    }
    return metrics


def render_dashboard(df: pd.DataFrame) -> None:
    st.set_page_config(page_title="Marketing Insights", page_icon="📈", layout="wide")
    apply_dashboard_theme()

    st.title("Sales and Marketing Insights")
    st.caption("Customer behavior, campaign performance, and purchase trends from the marketing dataset")

    metrics = get_overview_metrics(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{metrics['customers']:,}")
    with col2:
        st.metric("Avg Income", f"${metrics['avg_income']:,.0f}")
    with col3:
        st.metric("Avg Recency", f"{metrics['avg_recency']:.1f} days")
    with col4:
        st.metric("Campaign Response", f"{metrics['campaign_response_rate']:.1f}%")

    st.markdown('<div class="section-label">Customer profile</div>', unsafe_allow_html=True)
    left_col, right_col = st.columns(2)
    with left_col:
        st.plotly_chart(build_age_distribution(df), use_container_width=True)
    with right_col:
        st.plotly_chart(build_country_distribution(df), use_container_width=True)

    st.markdown('<div class="section-label">Campaign performance</div>', unsafe_allow_html=True)
    st.plotly_chart(build_campaign_chart(df), use_container_width=True)

    st.markdown('<div class="section-label">Purchase and spend behavior</div>', unsafe_allow_html=True)
    st.plotly_chart(build_spending_by_education(df), use_container_width=True)

    st.markdown('<div class="section-label">Predictive analysis</div>', unsafe_allow_html=True)
    linear_fig, regression_metrics = build_regression_summary(df)
    st.plotly_chart(linear_fig, use_container_width=True)

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(
            """
            <div class="insight-card">
                <strong>Key takeaway:</strong><br>
                The age profile shows the strongest concentration in mid-life customers, which is a valuable segment for retention and targeted offers.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col2:
        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Model performance:</strong><br>
                Regression MSE: <b>{regression_metrics['mse']:.2f}</b><br>
                R²: <b>{regression_metrics['r2']:.4f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    data_path = resolve_data_path()
    st.write(f"Loading dataset from: {data_path}")
    marketing = pd.read_csv(data_path)
    marketing = clean_marketing_data(marketing)

    render_dashboard(marketing)

    pio.write_html(build_age_distribution(marketing), "age_distribution.html", auto_open=False)
    pio.write_html(build_campaign_chart(marketing), "campaign_offers.html", auto_open=False)
    pio.write_html(build_spending_by_education(marketing), "education_spending.html", auto_open=False)
    pio.write_html(build_regression_summary(marketing)[0], "recency_regression.html", auto_open=False)


if __name__ == "__main__":
    main()
