import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("train.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday

def day_period(h):
    if 0 <= h < 6:
        return "night"
    elif 6 <= h < 12:
        return "morning"
    elif 12 <= h < 18:
        return "afternoon"
    else:
        return "evening"

df["day_period"] = df["hour"].apply(day_period)

# -----------------------------
# Sidebar widgets (INTERACTIVE)
# -----------------------------
st.sidebar.title("Filters")

year_filter = st.sidebar.multiselect(
    "Select Year",
    options=sorted(df["year"].unique()),
    default=sorted(df["year"].unique())
)

season_filter = st.sidebar.multiselect(
    "Select Season",
    options=sorted(df["season"].unique()),
    default=sorted(df["season"].unique())
)

workingday_filter = st.sidebar.radio(
    "Working Day",
    options=["All", "Working", "Non-working"]
)

hour_range = st.sidebar.slider(
    "Select Hour Range",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

# -----------------------------
# Apply filters
# -----------------------------
filtered_df = df[
    (df["year"].isin(year_filter)) &
    (df["season"].isin(season_filter)) &
    (df["hour"].between(hour_range[0], hour_range[1]))
]

if workingday_filter != "All":
    filtered_df = filtered_df[
        filtered_df["workingday"] == (1 if workingday_filter == "Working" else 0)
    ]

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("Bike Sharing Data – Interactive Dashboard")
st.write("Summary dashboard based on Assignments 3.1 and 3.2")

# -----------------------------
# Plot 1: Mean rentals by hour
# -----------------------------
st.subheader("Mean Rentals by Hour")
hourly_mean = filtered_df.groupby("hour")["count"].mean()

fig, ax = plt.subplots()
ax.plot(hourly_mean.index, hourly_mean.values)
ax.set_xlabel("Hour")
ax.set_ylabel("Mean Rentals")
st.pyplot(fig)

# -----------------------------
# Plot 2: Mean rentals by month
# -----------------------------
st.subheader("Mean Rentals by Month")
monthly_mean = filtered_df.groupby("month")["count"].mean()

fig, ax = plt.subplots()
ax.plot(monthly_mean.index, monthly_mean.values, marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Mean Rentals")
st.pyplot(fig)

# -----------------------------
# Plot 3: Working vs Non-working days
# -----------------------------
st.subheader("Working vs Non-working Days")
work_mean = filtered_df.groupby("workingday")["count"].mean()

fig, ax = plt.subplots()
ax.bar(["Non-working", "Working"], work_mean.values)
ax.set_ylabel("Mean Rentals")
st.pyplot(fig)

# -----------------------------
# Plot 4: Day period comparison
# -----------------------------
st.subheader("Mean Rentals by Day Period")
dp_mean = filtered_df.groupby("day_period")["count"].mean()

fig, ax = plt.subplots()
ax.bar(dp_mean.index, dp_mean.values)
ax.set_ylabel("Mean Rentals")
st.pyplot(fig)

# -----------------------------
# Plot 5: Weather impact
# -----------------------------
st.subheader("Mean Rentals by Weather")
weather_mean = filtered_df.groupby("weather")["count"].mean()

fig, ax = plt.subplots()
ax.bar(weather_mean.index, weather_mean.values)
ax.set_xlabel("Weather Category")
ax.set_ylabel("Mean Rentals")
st.pyplot(fig)

# -----------------------------
# Plot 6: Correlation heatmap
# -----------------------------
st.subheader("Correlation Heatmap")
num_cols = filtered_df.select_dtypes(include=np.number)
corr = num_cols.corr()

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)
