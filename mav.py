import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Maven Marketing Dashboard", page_icon="📊", layout="wide")


def load_data():
    df = pd.read_csv("marketing_data.csv", encoding="utf-8-sig")
    df.columns = [str(col).strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    if "Income" in df.columns:
        df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")

    for col in df.columns:
        if col in {"ID", "Year_Birth", "Kidhome", "Teenhome", "Recency", "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Response", "Complain"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data

def get_data():
    return load_data()


def format_currency(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


# Load and prepare data
raw_df = get_data()
df = raw_df.copy()

st.title("Maven Marketing Dashboard")
st.caption("Customer behavior, product spending, and campaign performance")

# Sidebar filters
st.sidebar.header("Filters")

country_options = sorted(df["Country"].dropna().unique().tolist()) if "Country" in df.columns else []
education_options = sorted(df["Education"].dropna().unique().tolist()) if "Education" in df.columns else []
marital_options = sorted(df["Marital_Status"].dropna().unique().tolist()) if "Marital_Status" in df.columns else []

selected_country = st.sidebar.multiselect("Country", country_options)
selected_education = st.sidebar.multiselect("Education", education_options)
selected_marital = st.sidebar.multiselect("Marital Status", marital_options)

if selected_country:
    df = df[df["Country"].isin(selected_country)]
if selected_education:
    df = df[df["Education"].isin(selected_education)]
if selected_marital:
    df = df[df["Marital_Status"].isin(selected_marital)]

st.subheader("Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Customers", f"{len(df):,}")
kpi2.metric("Average Income", format_currency(df["Income"].mean()) if "Income" in df.columns else "N/A")
kpi3.metric("Average Total Spend", format_currency(df[[c for c in df.columns if c.startswith("Mnt")]].sum(axis=1).mean()) if any(c.startswith("Mnt") for c in df.columns) else "N/A")
kpi4.metric("Response Rate", f"{(df['Response'].mean() * 100):.1f}%" if "Response" in df.columns else "N/A")

# Product sales by category
product_cols = [c for c in df.columns if c.startswith("Mnt")]
if product_cols:
    product_spend = df[product_cols].sum().sort_values(ascending=False)

    col_left, col_right = st.columns([1.5, 1])
    with col_left:
        st.subheader("Spend by Product Category")
        st.bar_chart(product_spend)

    with col_right:
        st.subheader("Top Categories")
        for item, value in product_spend.head(6).items():
            st.markdown(f"- **{item.replace('Mnt', '').replace('Products', '')}**: {format_currency(value)}")

# Campaign performance
campaign_cols = [c for c in df.columns if c.startswith("AcceptedCmp")]
if campaign_cols:
    st.subheader("Campaign Acceptance")
    campaign_acceptance = df[campaign_cols].sum().sort_index()
    st.bar_chart(campaign_acceptance)

# Customer demographics
st.subheader("Customer Demographics")

fig, ax = plt.subplots(figsize=(10, 5))
if "Education" in df.columns:
    sns.countplot(data=df, x="Education", ax=ax, order=df["Education"].dropna().value_counts().index)
    ax.set_title("Customers by Education Level")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

# Sales by country
if "Country" in df.columns:
    st.subheader("Top Countries by Customer Count")
    country_counts = df["Country"].dropna().value_counts().head(10)
    st.bar_chart(country_counts)

# Data preview
st.subheader("Data Preview")
st.dataframe(df.head(20), use_container_width=True)

# Optional metrics row
with st.expander("Dataset summary"):
    st.write(df.describe(include="all").T)

st.caption("Built for deployment on Streamlit Cloud")
