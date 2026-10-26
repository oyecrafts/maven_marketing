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

BRAND_COLORS = {
    "navy": "#0f172a",
    "slate": "#1e293b",
    "card": "#111827",
    "violet": "#8b5cf6",
    "emerald": "#34d399",
    "blue": "#60a5fa",
    "amber": "#fbbf24",
    "pink": "#f472b6",
}


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
                --bg-soft: #111827;
                --text: #f8fafc;
                --muted: #cbd5e1;
                --card: rgba(15, 23, 42, 0.82);
                --border: rgba(148, 163, 184, 0.2);
                --violet: #8b5cf6;
                --emerald: #34d399;
                --blue: #60a5fa;
                --amber: #fbbf24;
                --pink: #f472b6;
            }

            .stApp {
                background: linear-gradient(135deg, #020817 0%, #0f172a 35%, #111827 100%);
                color: var(--text);
            }

            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3, p, div, label, span {
                color: var(--text) !important;
            }

            .stMetric {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 0.9rem;
                padding: 0.8rem 1rem;
                box-shadow: 0 10px 25px rgba(2, 8, 23, 0.25);
            }

            div[data-testid="stMetricValue"] {
                font-size: 1.6rem;
                font-weight: 700;
                color: #ffffff !important;
            }

            div[data-testid="stMetricLabel"] {
                color: #cbd5e1 !important;
                font-size: 0.78rem;
                letter-spacing: 0.03em;
            }

            .section-header {
                font-size: 0.72rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #c4b5fd !important;
                margin-top: 1rem;
                margin-bottom: 0.6rem;
                font-weight: 700;
            }

            .insight-card {
                background: rgba(15, 23, 42, 0.92);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 1rem;
                padding: 1rem 1.1rem;
                min-height: 110px;
                box-shadow: inset 0 0 0 1px rgba(139, 92, 246, 0.15);
            }

            .stSidebar {
                background: rgba(15, 23, 42, 0.9);
            }

            [data-testid="stSidebarNav"] {
                background: rgba(15, 23, 42, 0.9);
            }

            .stSelectbox, .stMultiSelect, .stSlider label, .stRadio label {
                color: #f8fafc !important;
            }

            .stDownloadButton > button {
                background: linear-gradient(90deg, var(--violet), #7c3aed);
                border: 1px solid rgba(255,255,255,0.18);
                color: white !important;
                font-weight: 600;
                border-radius: 0.7rem;
            }

            .stButton > button {
                background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
                color: white;
                border-radius: 0.7rem;
                border: none;
            }

            div[data-testid="stPlotlyChart"] {
                overflow: visible !important;
            }

            .js-plotly-plot .plotly .legend {
                overflow: visible !important;
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
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=15, r=15, t=45, b=20),
        )
        return empty

    min_age = float(df["Age"].min())
    max_age = float(df["Age"].max())
    age_bins = [min_age, 30, 40, 50, 60, max_age]
    age_labels = ["<30", "30-39", "40-49", "50-59", "60+"]
    df_plot = df.copy()
    df_plot["Age_group"] = pd.cut(df_plot["Age"], bins=age_bins, labels=age_labels, right=False)
    counts = df_plot["Age_group"].value_counts().sort_index()

    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=BRAND_COLORS["pink"]))
    fig.update_layout(
        title="Age Distribution",
        xaxis_title="Age Group",
        yaxis_title="Customers",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=45, b=20),
        showlegend=False,
        font=dict(color="#f8fafc"),
        height=420,
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
                color=[
                    BRAND_COLORS["blue"],
                    BRAND_COLORS["emerald"],
                    BRAND_COLORS["amber"],
                    BRAND_COLORS["pink"],
                    BRAND_COLORS["violet"],
                ]
            ),
        )
    )
    fig.update_layout(
        title="Campaign Offer Acceptance",
        xaxis_title="Campaign",
        yaxis_title="Responses",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=45, b=20),
        height=420,
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
        return px.bar(title="Spending by Education Level")

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
        color_discrete_sequence=[
            BRAND_COLORS["blue"],
            BRAND_COLORS["emerald"],
            BRAND_COLORS["amber"],
            BRAND_COLORS["pink"],
            BRAND_COLORS["violet"],
            "#f87171",
        ],
        labels={
            "Education": "Education Level",
            "Amount": "Total Spend",
            "Product": "Product Category",
        },
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=100, b=30),
        height=520,
        legend={
            "title": {"text": "Product Category", "font": {"color": "#f8fafc", "size": 14}},
            "orientation": "h",
            "x": 0.0,
            "y": 1.18,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(15, 23, 42, 0.45)",
            "bordercolor": "rgba(148, 163, 184, 0.12)",
            "borderwidth": 1,
            "font": {"size": 12, "color": "#f8fafc"},
            "traceorder": "normal",
        },
        xaxis={
            "title": {"text": "Education Level", "font": {"size": 15, "color": "#f8fafc"}},
            "tickfont": {"size": 12, "color": "#e2e8f0"},
        },
        yaxis={
            "title": {"text": "Total Spend", "font": {"size": 15, "color": "#f8fafc"}},
            "tickfont": {"size": 12, "color": "#e2e8f0"},
        },
    )
    return fig


def build_country_distribution(df: pd.DataFrame) -> go.Figure:
    if "Country" not in df.columns:
        empty = go.Figure()
        empty.update_layout(
            title="Top Customer Countries",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=15, r=15, t=45, b=20),
        )
        return empty

    country_counts = df["Country"].value_counts().head(8).sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation="h",
            marker_color=BRAND_COLORS["violet"],
        )
    )
    fig.update_layout(
        title="Top Customer Countries",
        xaxis_title="Customers",
        yaxis_title="Country",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=45, b=20),
        showlegend=False,
        height=420,
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
            marker=dict(color=BRAND_COLORS["violet"], size=10, opacity=0.8),
        )
    )
    max_val = max(float(y_test.max()), float(y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Perfect Predictions",
            line=dict(color=BRAND_COLORS["amber"], dash="dash"),
        )
    )
    fig.update_layout(
        title="Recency Prediction Performance",
        xaxis_title="Actual Recency",
        yaxis_title="Predicted Recency",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
    )

    return fig, {"mse": mse, "r2": r2}


def get_overview_metrics(df: pd.DataFrame) -> dict:
    return {
        "customers": len(df),
        "avg_income": float(df["Income"].mean()) if "Income" in df.columns else 0.0,
        "avg_recency": float(df["Recency"].mean()) if "Recency" in df.columns else 0.0,
        "campaign_response_rate": (
            float(df["Response"].mean() * 100) if "Response" in df.columns else 0.0
        ),
    }


def get_filtered_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    if "Education" in filtered.columns:
        education_options = sorted(filtered["Education"].dropna().unique().tolist())
        selected_education = st.sidebar.multiselect(
            "Education",
            education_options,
            default=education_options,
        )
        if selected_education:
            filtered = filtered[filtered["Education"].isin(selected_education)]

    if "Country" in filtered.columns:
        country_options = sorted(filtered["Country"].dropna().unique().tolist())
        selected_countries = st.sidebar.multiselect(
            "Country",
            country_options,
            default=country_options,
        )
        if selected_countries:
            filtered = filtered[filtered["Country"].isin(selected_countries)]

    if "Marital_Status" in filtered.columns:
        marital_options = sorted(filtered["Marital_Status"].dropna().unique().tolist())
        selected_status = st.sidebar.multiselect(
            "Marital Status",
            marital_options,
            default=marital_options,
        )
        if selected_status:
            filtered = filtered[filtered["Marital_Status"].isin(selected_status)]

    if "Age" in filtered.columns:
        age_min = int(filtered["Age"].min())
        age_max = int(filtered["Age"].max())
        age_range = st.sidebar.slider(
            "Age range",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            step=1,
        )
        filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]

    if "Income" in filtered.columns:
        income_min = int(filtered["Income"].min())
        income_max = int(filtered["Income"].max())
        income_range = st.sidebar.slider(
            "Income range",
            min_value=income_min,
            max_value=income_max,
            value=(income_min, income_max),
            step=1000,
        )
        filtered = filtered[
            (filtered["Income"] >= income_range[0]) & (filtered["Income"] <= income_range[1])
        ]

    return filtered


def build_dashboard_snapshot_html(df: pd.DataFrame) -> str:
    age_chart = build_age_distribution(df)
    campaign_chart = build_campaign_chart(df)
    spend_chart = build_spending_by_education(df)
    regression_chart, _ = build_regression_summary(df)

    charts = [
        pio.to_html(age_chart, full_html=False, include_plotlyjs="cdn"),
        pio.to_html(campaign_chart, full_html=False, include_plotlyjs="cdn"),
        pio.to_html(spend_chart, full_html=False, include_plotlyjs="cdn"),
        pio.to_html(regression_chart, full_html=False, include_plotlyjs="cdn"),
    ]

    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Marketing Dashboard Snapshot</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            padding: 24px;
          }}
          h1 {{ color: #f8fafc; }}
          .card {{
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 20px;
          }}
        </style>
      </head>
      <body>
        <h1>Marketing Dashboard Snapshot</h1>
        <div class="card">{charts[0]}</div>
        <div class="card">{charts[1]}</div>
        <div class="card">{charts[2]}</div>
        <div class="card">{charts[3]}</div>
      </body>
    </html>
    """


def render_dashboard(df: pd.DataFrame) -> None:
    st.set_page_config(page_title="Marketing Insights", page_icon="📈", layout="wide")
    apply_dashboard_theme()

    st.title("Sales and Marketing Insights")
    st.caption("Customer behavior, campaign response, spend patterns, and predictive recency analysis")

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

    st.markdown('<div class="section-header">Customer profile</div>', unsafe_allow_html=True)
    left_col, right_col = st.columns(2)
    with left_col:
        st.plotly_chart(build_age_distribution(df), use_container_width=True)
    with right_col:
        st.plotly_chart(build_country_distribution(df), use_container_width=True)

    st.markdown('<div class="section-header">Campaign performance</div>', unsafe_allow_html=True)
    st.plotly_chart(build_campaign_chart(df), use_container_width=True)

    st.markdown('<div class="section-header">Purchase and spend behavior</div>', unsafe_allow_html=True)
    st.plotly_chart(build_spending_by_education(df), use_container_width=True)

    st.markdown('<div class="section-header">Predictive analysis</div>', unsafe_allow_html=True)
    linear_fig, regression_metrics = build_regression_summary(df)
    st.plotly_chart(linear_fig, use_container_width=True)

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown(
            """
            <div class="insight-card">
                <strong>Key takeaway</strong><br>
                The strongest concentration of customers sits in the mid-life age bands, which is a key opportunity for retention and personalized campaigns.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col2:
        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Model performance</strong><br>
                MSE: <b>{regression_metrics['mse']:.2f}</b><br>
                R²: <b>{regression_metrics['r2']:.4f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col3:
        st.markdown(
            """
            <div class="insight-card">
                <strong>Recommended action</strong><br>
                Prioritize higher-value education segments and campaign offers that drive repeat purchases and customer re-engagement.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Exports</div>', unsafe_allow_html=True)
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            label="Download dashboard snapshot (.html)",
            data=build_dashboard_snapshot_html(df),
            file_name="marketing_dashboard_snapshot.html",
            mime="text/html",
        )
    with export_col2:
        st.download_button(
            label="Download filtered data (.csv)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_marketing_data.csv",
            mime="text/csv",
        )


def main() -> None:
    data_path = resolve_data_path()
    st.write(f"Loading dataset from: {data_path}")
    marketing = pd.read_csv(data_path)
    marketing = clean_marketing_data(marketing)

    filtered = get_filtered_dataframe(marketing)

    if filtered.empty:
        st.warning("No data matches the selected filters. Please broaden the filters and try again.")
        return

    render_dashboard(filtered)

    pio.write_html(build_age_distribution(filtered), "age_distribution.html", auto_open=False)
    pio.write_html(build_campaign_chart(filtered), "campaign_offers.html", auto_open=False)
    pio.write_html(build_spending_by_education(filtered), "education_spending.html", auto_open=False)
    pio.write_html(build_regression_summary(filtered)[0], "recency_regression.html", auto_open=False)


if __name__ == "__main__":
    main()
