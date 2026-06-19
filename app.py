# Pakistan Economic Indicators Dashboard
# Data: Kaggle (pakistan_economic_indicators_2000_2025)

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Pakistan Economy",
    page_icon="📈",
    layout="wide"
)

# load data and cache it so it doesn't reload on every interaction
@st.cache_data
def load_data():
    path = os.path.join("data", "pakistan_economic_indicators_2000_2025.csv")
    df = pd.read_csv(path)
    return df

df = load_data()

# app title and description
st.title("PK Pakistan Economic Indicators (2000-2025)")
st.markdown("Analyzing Pakistan's key economic trends over 25 years")

# show raw data in a collapsible section
with st.expander("Show Raw Data"):
    st.dataframe(df)
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

# quick data overview
st.subheader("Data Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Years Covered", f"{int(df['year'].min())} - {int(df['year'].max())}")

with col2:
    st.metric("Highest Inflation", f"{df['inflation_cpi_pct'].max()}%")

with col3:
    st.metric("Avg GDP Growth", f"{df['gdp_growth_pct'].mean():.1f}%")

with col4:
    st.metric("Latest PKR/USD", f"{df['pkr_per_usd'].iloc[-1]}")

st.markdown("---")

# sidebar - user can select which indicator to explore
st.sidebar.title("Filters")
indicator = st.sidebar.selectbox(
    "Select Indicator",
    ["inflation_cpi_pct", "gdp_growth_pct", "unemployment_pct", 
     "pkr_per_usd", "exports_usd_bn", "imports_usd_bn"]
)

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(2000, 2025)
)

# filter data based on year range
filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# main line chart
st.subheader(f"{indicator.replace('_', ' ').title()} Over Time")

fig = px.line(
    filtered_df,
    x="year",
    y=indicator,
    markers=True,
    title=f"Pakistan {indicator.replace('_', ' ').title()} (2000-2025)"
)

fig.update_layout(
    xaxis_title="Year",
    yaxis_title=indicator,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# GDP and Inflation comparison - bar chart
st.subheader("GDP Growth vs Inflation")

fig2 = px.bar(
    filtered_df,
    x="year",
    y=["gdp_growth_pct", "inflation_cpi_pct"],
    barmode="group",
    title="GDP Growth vs Inflation Rate"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# PKR depreciation over time
st.subheader("PKR/USD Exchange Rate")

fig3 = px.area(
    filtered_df,
    x="year",
    y="pkr_per_usd",
    title="Pakistani Rupee Depreciation Against USD",
    color_discrete_sequence=["#ef4444"]
)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Exports vs Imports
st.subheader("Exports vs Imports (Billion USD)")

fig4 = px.line(
    filtered_df,
    x="year",
    y=["exports_usd_bn", "imports_usd_bn"],
    markers=True,
    title="Pakistan Exports vs Imports"
)

st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# correlation heatmap - shows how indicators relate to each other
st.subheader("How Economic Indicators Relate to Each Other")

numeric_cols = ["gdp_growth_pct", "inflation_cpi_pct", "unemployment_pct", 
                "pkr_per_usd", "exports_usd_bn", "imports_usd_bn"]

corr = df[numeric_cols].corr()

fig5 = px.imshow(
    corr,
    title="Correlation Heatmap",
    color_continuous_scale="RdBu_r",
    aspect="auto"
)

st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ── ML: INFLATION PREDICTOR ─────────────────────────────
st.subheader("🔮 Predict Future Inflation")
st.markdown("Using Linear Regression to predict inflation based on historical trend")

# prepare data for the model
# X = input (year), y = output (inflation we want to predict)
X = df[['year']]
y = df['inflation_cpi_pct']

# create and train the model
model = LinearRegression()
model.fit(X, y)

# let user pick a future year
future_year = st.slider("Select a year to predict", min_value=2026, max_value=2030, value=2026)

# make prediction
predicted_inflation = model.predict([[future_year]])[0]

st.metric(f"Predicted Inflation for {future_year}", f"{predicted_inflation:.2f}%")

st.info("⚠️ Note: This is a simple linear trend prediction. Real inflation depends on many factors (policy, global events, etc.) that this basic model doesn't account for.")

st.markdown("---")

# key insights section
st.subheader("Key Insights")

col1, col2 = st.columns(2)

with col1:
    worst_inflation_year = df.loc[df['inflation_cpi_pct'].idxmax(), 'year']
    st.info(f"📌 Highest inflation was in **{int(worst_inflation_year)}** at **{df['inflation_cpi_pct'].max()}%**")
    
    best_gdp_year = df.loc[df['gdp_growth_pct'].idxmax(), 'year']
    st.success(f"📌 Best GDP growth was in **{int(best_gdp_year)}** at **{df['gdp_growth_pct'].max()}%**")

with col2:
    pkr_drop = df['pkr_per_usd'].max() - df['pkr_per_usd'].min()
    st.warning(f"📌 PKR dropped by **{pkr_drop:.1f} rupees** against USD since 2000")
    
    trade_deficit = df['imports_usd_bn'].sum() - df['exports_usd_bn'].sum()
    st.error(f"📌 Total trade deficit over 25 years: **${trade_deficit:.1f}B**")

st.markdown("---")

# footer
st.markdown("**Data Source:** Kaggle - Pakistan Economic Indicators 2000-2025")
st.markdown("**Built with:** Python, Streamlit, Plotly, Pandas, Scikit-learn")