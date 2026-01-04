import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Sales Dashboard",
    layout="wide"
)

st.title("📊 Sales Data Analysis Dashboard")
st.markdown("Interactive & Colourful Dashboard using Streamlit")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("D:\Python Files\sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🔎 Filter Options")

region_filter = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

product_filter = st.sidebar.multiselect(
    "Select Product",
    options=df["Product"].unique(),
    default=df["Product"].unique()
)

filtered_df = df[
    (df["Region"].isin(region_filter)) &
    (df["Product"].isin(product_filter))
]

# KPI Section
total_sales = filtered_df["Total_Sales"].sum()
total_quantity = filtered_df["Quantity"].sum()
avg_price = filtered_df["Price"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales", f"₹ {total_sales:,.0f}")
col2.metric("📦 Total Quantity", total_quantity)
col3.metric("🏷️ Average Price", f"₹ {avg_price:,.0f}")

st.divider()

# Charts Section
col4, col5 = st.columns(2)

# Sales by Region
with col4:
    st.subheader("📍 Sales by Region")
    sales_by_region = filtered_df.groupby("Region")["Total_Sales"].sum()

    fig1, ax1 = plt.subplots()
    sales_by_region.plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Total Sales")
    st.pyplot(fig1)

# Sales by Product
with col5:
    st.subheader("🛒 Sales by Product")
    sales_by_product = filtered_df.groupby("Product")["Total_Sales"].sum()

    fig2, ax2 = plt.subplots()
    sales_by_product.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Product")
    ax2.set_ylabel("Total Sales")
    st.pyplot(fig2)

st.divider()

# Monthly Trend
st.subheader("📈 Monthly Sales Trend")
monthly_sales = filtered_df.groupby(
    filtered_df["Date"].dt.to_period("M")
)["Total_Sales"].sum()

fig3, ax3 = plt.subplots()
monthly_sales.plot(ax=ax3)
ax3.set_xlabel("Month")
ax3.set_ylabel("Total Sales")
st.pyplot(fig3)

st.divider()

# Raw Data
with st.expander("📄 View Raw Data"):
    st.dataframe(filtered_df)