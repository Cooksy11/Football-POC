# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:33:51 2025

@author: Sukhdeep.Sangha
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# === Load Data ===
st.title("Football Expectancy Tracker")
st.markdown("Tracks EXP changes versus baseline EXP across selected filters.")

# Load data from Google Sheets CSV export
sheet_url = st.secrets["csv_url"]  # stored in .streamlit/secrets.toml
@st.cache_data

def load_data():
    df = pd.read_csv(sheet_url)
    df["MINUTES"] = pd.to_numeric(df["MINUTES"], errors="coerce")
    df = df.dropna(subset=["SRC_EVENT_ID", "MINUTES"])
    df.sort_values(["SRC_EVENT_ID", "MINUTES"], inplace=True)
    return df

df = load_data()

# === Determine Favourites and Baselines ===
first_rows = df.groupby("SRC_EVENT_ID").first().reset_index()
fav_map = {}
fav_exp_map, dog_exp_map, total_exp_map = {}, {}, {}
for _, row in first_rows.iterrows():
    fav = "Home" if row["GOAL_EXP_HOME"] > row["GOAL_EXP_AWAY"] else "Away"
    fav_map[row["SRC_EVENT_ID"]] = fav
    fav_exp = row["GOAL_EXP_HOME"] if fav == "Home" else row["GOAL_EXP_AWAY"]
    dog_exp = row["GOAL_EXP_AWAY"] if fav == "Home" else row["GOAL_EXP_HOME"]
    fav_exp_map[row["SRC_EVENT_ID"]] = fav_exp
    dog_exp_map[row["SRC_EVENT_ID"]] = dog_exp
    total_exp_map[row["SRC_EVENT_ID"]] = row["GOAL_EXP_HOME"] + row["GOAL_EXP_AWAY"]

df["Favourite"] = df["SRC_EVENT_ID"].map(fav_map)
df["Fav_EXP"] = np.where(df["Favourite"] == "Home", df["GOAL_EXP_HOME"], df["GOAL_EXP_AWAY"])
df["Dog_EXP"] = np.where(df["Favourite"] == "Home", df["GOAL_EXP_AWAY"], df["GOAL_EXP_HOME"])
df["Total_EXP"] = df["GOAL_EXP_HOME"] + df["GOAL_EXP_AWAY"]
df["Base_Fav_EXP"] = df["SRC_EVENT_ID"].map(fav_exp_map)
df["Base_Dog_EXP"] = df["SRC_EVENT_ID"].map(dog_exp_map)
df["Base_Total_EXP"] = df["SRC_EVENT_ID"].map(total_exp_map)
df["Change_Fav"] = df["Fav_EXP"] - df["Base_Fav_EXP"]
df["Change_Dog"] = df["Dog_EXP"] - df["Base_Dog_EXP"]
df["Change_Total"] = df["Total_EXP"] - df["Base_Total_EXP"]

# === Time Band ===
bins = list(range(0, 91, 5)) + [1000]
labels = [f"{i}-{i+5}" for i in range(0, 90, 5)] + ["85-90+"]
df["Time Band"] = pd.cut(df["MINUTES"], bins=bins, labels=labels, right=False)

# === Scoreline Classification ===
def scoreline(row):
    if row["Favourite"] == "Home" and row["GOALS_HOME"] > row["GOALS_AWAY"]:
        return "Favourite Winning"
    elif row["Favourite"] == "Away" and row["GOALS_AWAY"] > row["GOALS_HOME"]:
        return "Favourite Winning"
    elif row["GOALS_HOME"] == row["GOALS_AWAY"]:
        return "Scores Level"
    else:
        return "Underdog Winning"
df["Scoreline"] = df.apply(scoreline, axis=1)

# === Favouritism Level ===
first_rows["FAV_DIFF"] = abs(first_rows["GOAL_EXP_HOME"] - first_rows["GOAL_EXP_AWAY"])
def classify_fav(diff):
    if diff > 1:
        return "Strong"
    elif diff >= 0.5:
        return "Medium"
    else:
        return "Slight"
fav_level_map = dict(zip(first_rows["SRC_EVENT_ID"], first_rows["FAV_DIFF"].apply(classify_fav)))
df["Favouritism Level"] = df["SRC_EVENT_ID"].map(fav_level_map)

# === Goals Over/Under Performance ===
df["Goals_Total"] = df["GOALS_HOME"] + df["GOALS_AWAY"]

def expected_goals_so_far(row, who):
    minute = row["MINUTES"]
    if minute <= 45:
        perc = row["FIRST_HALF_GOALS_PERC"] * (minute / 45)
    else:
        perc = row["FIRST_HALF_GOALS_PERC"] + (1 - row["FIRST_HALF_GOALS_PERC"]) * ((minute - 45) / 45)
    base = total_exp_map[row["SRC_EVENT_ID"]] if who == "Total" else fav_exp_map[row["SRC_EVENT_ID"]] if who == "Fav" else dog_exp_map[row["SRC_EVENT_ID"]]
    return perc * base

for who in ["Fav", "Dog", "Total"]:
    df[f"{who}_EXP_SO_FAR"] = df.apply(lambda row: expected_goals_so_far(row, who), axis=1)
    df[f"{who}_Goals_Performance"] = df["Goals_Total"] - df[f"{who}_EXP_SO_FAR"]
    df[f"{who}_Perf_Label"] = np.where(df[f"{who}_Goals_Performance"] > 0, f"{who} Over Performance", f"{who} Under Performance")

# === Filters ===
st.sidebar.header("Filter Options")
exp_options = st.sidebar.multiselect("Choose EXP View(s):", ["Change_Total", "Change_Fav", "Change_Dog"], default=["Change_Total"])
scoreline_filter = st.sidebar.multiselect("Scoreline:", df["Scoreline"].unique(), default=list(df["Scoreline"].unique()))
favlevel_filter = st.sidebar.multiselect("Favouritism Level:", df["Favouritism Level"].unique(), default=list(df["Favouritism Level"].unique()))
goals_perf_filter = st.sidebar.multiselect("Goals Performance:",
    ["Fav Over Performance", "Fav Under Performance", "Dog Over Performance", "Dog Under Performance", "Total Over Performance", "Total Under Performance"],
    default=[])

# === Plot Charts ===
for exp_col in exp_options:
    filtered = df[df["Scoreline"].isin(scoreline_filter) & df["Favouritism Level"].isin(favlevel_filter)]
    if goals_perf_filter:
        if "Change_Total" in exp_col:
            filtered = filtered[filtered["Total_Perf_Label"].isin(goals_perf_filter)]
        elif "Fav" in exp_col:
            filtered = filtered[filtered["Fav_Perf_Label"].isin(goals_perf_filter)]
        elif "Dog" in exp_col:
            filtered = filtered[filtered["Dog_Perf_Label"].isin(goals_perf_filter)]

    avg_change = filtered.groupby("Time Band")[exp_col].mean()
    match_counts = filtered.groupby("Time Band")["SRC_EVENT_ID"].nunique()
    bands = df["Time Band"].cat.categories
    avg_change = avg_change.reindex(bands, fill_value=np.nan)
    match_counts = match_counts.reindex(bands, fill_value=0)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(avg_change.index, avg_change.values, color='black', marker='o')
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.set_ylabel("Avg EXP Change")
    ax1.set_xlabel("Time Band (Minutes)")
    ax1.set_xticks(range(len(bands)))
    ax1.set_xticklabels(bands, rotation=45)
    ax2 = ax1.twinx()
    ax2.bar(match_counts.index, match_counts.values, width=0.6, color='skyblue', alpha=0.4)
    ax2.set_ylabel("# Unique Matches")
    plt.title(f"{exp_col.replace('_', ' ')} vs Time")
    plt.grid(True, axis='y')
    st.pyplot(fig)
