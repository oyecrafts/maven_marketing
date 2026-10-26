from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, dcc, html
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


def build_age_distribution(df: pd.DataFrame) -> go.Figure:
    min_age = float(df["Age"].min()) if "Age" in df.columns else 0
    max_age = float(df["Age"].max()) if "Age" in df.columns else 0
    age_bins = [min_age, 30, 40, 50, 60, max_age]
    age_labels = ["<30", "30-39", "40-49", "50-59", "60+"]
    df = df.copy()
    df["Age_group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)
    counts = df["Age_group"].value_counts().sort_index()

    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="lightcoral"))
    fig.update_layout(
        title="Distribution of Age Groups",
        xaxis_title="Age Group",
        yaxis_title="Number of Individuals",
        xaxis=dict(tickangle=45),
        yaxis=dict(gridcolor="lightgrey"),
        plot_bgcolor="rgba(0,0,0,0)",
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
            marker=dict(color=["skyblue", "lightgreen", "lightcoral", "lightsalmon", "lightyellow"]),
        )
    )
    fig.update_layout(
        title="Acceptance Rates of Campaign Offers",
        xaxis=dict(title="Campaign"),
        yaxis=dict(title="Number of Responses"),
    )
    return fig


def build_spending_by_education(df: pd.DataFrame) -> px.bar:
    product_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    education_spending = df.groupby("Education")[product_columns].sum().reset_index()
    melted = education_spending.melt(id_vars="Education", var_name="Product", value_name="Amount")
    fig = px.bar(
        melted,
        x="Education",
        y="Amount",
        color="Product",
        labels={"Education": "Education Level", "Amount": "Amount Spent", "Product": "Product Category"},
        title="Amount Spent on Different Product Categories by Education Level",
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
            marker=dict(color="purple", size=10, opacity=0.8),
        )
    )
    max_val = max(float(y_test.max()), float(y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Perfect Predictions",
            line=dict(color="gold", dash="dash"),
        )
    )
    fig.update_layout(
        title="Actual vs. Predicted Recency",
        xaxis_title="Actual Recency",
        yaxis_title="Predicted Recency",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(x=0.8, y=0.1),
    )

    return fig, {"mse": mse, "r2": r2}


def create_dashboard(df: pd.DataFrame) -> Dash:
    age_fig = build_age_distribution(df)
    campaign_fig = build_campaign_chart(df)
    education_spending_fig = build_spending_by_education(df)
    linear_fig, metrics = build_regression_summary(df)

    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#111111", "padding": "20px", "color": "white"},
        children=[
            html.H1("Sales and Marketing Insights", style={"textAlign": "center"}),
            html.Div(
                children=[
                    html.Div(
                        html.Div(
                            [
                                html.H3(f"Regression MSE: {metrics['mse']:.2f}"),
                                html.H3(f"Regression R²: {metrics['r2']:.4f}"),
                            ],
                            style={"padding": "15px", "border": "1px solid #444", "borderRadius": "8px"},
                        ),
                        style={"marginBottom": "20px"},
                    )
                ]
            ),
            html.Div(
                [
                    html.Div([dcc.Graph(figure=age_fig, config={"displayModeBar": False})], style={"width": "48%", "display": "inline-block"}),
                    html.Div([dcc.Graph(figure=campaign_fig, config={"displayModeBar": False})], style={"width": "48%", "display": "inline-block", "marginLeft": "2%"}),
                ]
            ),
            html.Div([dcc.Graph(figure=education_spending_fig, config={"displayModeBar": False})], style={"marginTop": "20px"}),
            html.Div([dcc.Graph(figure=linear_fig, config={"displayModeBar": False})], style={"marginTop": "20px"}),
        ],
    )
    return app


def main() -> None:
    data_path = resolve_data_path()
    print(f"Loading dataset from: {data_path}")
    marketing = pd.read_csv(data_path)
    marketing = clean_marketing_data(marketing)

    print(marketing.head())
    print(marketing.columns.tolist())
    print(marketing.describe(include="all").head())

    fig_age = build_age_distribution(marketing)
    fig_campaign = build_campaign_chart(marketing)
    fig_education = build_spending_by_education(marketing)
    fig_linear, metrics = build_regression_summary(marketing)

    pio.write_html(fig_age, "age_distribution.html", auto_open=False)
    pio.write_html(fig_campaign, "campaign_offers.html", auto_open=False)
    pio.write_html(fig_education, "education_spending.html", auto_open=False)
    pio.write_html(fig_linear, "recency_regression.html", auto_open=False)

    print(f"Regression model metrics: mse={metrics['mse']:.4f}, r2={metrics['r2']:.4f}")

    app = create_dashboard(marketing)
    app.run(debug=True, host="0.0.0.0", port=8051)


if __name__ == "__main__":
    main()
