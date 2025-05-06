import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import requests

# === Load Data from Google Sheets ===
# Make sure this is the CSV export link (not the editable sheet URL)
csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSY2HTi7cZbLFNcHHGUJub1dZszQhLRSRKsSzEiMdqPC0bbvU7j8EG6iWOkZ3zHpA/pub?output=csv"
response = requests.get(csv_url)
df = pd.read_csv(StringIO(response.text))

# === Preprocessing ===
time_col = "MINUTES"
event_col = "SRC_EVENT_ID"
home_score_col = "GOALS_HOME"
away_score_col = "GOALS_AWAY"

exp_definitions = {
    "Goals": {"home": "GOAL_EXP_HOME", "away": "GOAL_EXP_AWAY"},
    "Corners": {"home": "CORNERS_EXP_HOME", "away": "CORNERS_EXP_AWAY"},
    "Yellow Cards": {"home": "YELLOW_CARDS_EXP_HOME", "away": "YELLOW_CARDS_EXP_AWAY"}
}

# Drop invalid rows and sort by event time
df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
df.dropna(subset=[event_col, time_col], inplace=True)
df = df.sort_values([event_col, time_col])

# Determine favourites and favouritism level
first_rows = df.groupby(event_col).first().reset_index()
df["Favourite"] = df[event_col].map({
    row[event_col]: (
        "Home" if row["GOAL_EXP_HOME"] > row["GOAL_EXP_AWAY"]
        else "Away" if row["GOAL_EXP_AWAY"] > row["GOAL_EXP_HOME"]
        else "None"
    ) for _, row in first_rows.iterrows()
})
df["Favouritism_Level"] = df[event_col].map({
    row[event_col]: (
        "Strong" if abs(row["GOAL_EXP_HOME"] - row["GOAL_EXP_AWAY"]) > 1 else
        "Medium" if abs(row["GOAL_EXP_HOME"] - row["GOAL_EXP_AWAY"]) >= 0.5 else
        "Slight"
    ) for _, row in first_rows.iterrows()
})

# Compute Scoreline State
def scoreline(row):
    if row["Favourite"] == "Home" and row[home_score_col] > row[away_score_col]:
        return "Favourite Winning"
    elif row["Favourite"] == "Away" and row[away_score_col] > row[home_score_col]:
        return "Favourite Winning"
    elif row[home_score_col] == row[away_score_col]:
        return "Scores Level"
    else:
        return "Underdog Winning"
df["Scoreline"] = df.apply(scoreline, axis=1)

# Time bands
bins = list(range(0, 91, 5)) + [1000]
labels = [f"{i}-{i+5}" for i in range(0, 90, 5)] + ["85-90+"]
df["Time Band"] = pd.cut(df[time_col], bins=bins, labels=labels, right=False)

# === Streamlit App UI ===
st.set_page_config(layout="wide")
st.title("Football Expectancy Change Tracker")

exp_options = ["Goals", "Corners", "Yellow Cards"]
chart_types = ["Total", "Favourite", "Underdog"]

selected_exps = st.multiselect("Select EXP Charts to View", [f"{typ} {exp}" for exp in exp_options for typ in chart_types])

# Filters
scoreline_filter = st.multiselect("Scoreline State", ["Favourite Winning", "Underdog Winning", "Scores Level"])
fav_level_filter = st.multiselect("Level of Favouritism", ["Strong", "Medium", "Slight"])

# Filtered base
df_filtered = df.copy()
if scoreline_filter:
    df_filtered = df_filtered[df_filtered["Scoreline"].isin(scoreline_filter)]
if fav_level_filter:
    df_filtered = df_filtered[df_filtered["Favouritism_Level"].isin(fav_level_filter)]

# === Plotting ===
for selection in selected_exps:
    view_type, exp_name = selection.split()
    if exp_name not in exp_definitions:
        continue

    df_exp = df_filtered.copy()
    df_exp["Home"] = df_exp[exp_definitions[exp_name]["home"]]
    df_exp["Away"] = df_exp[exp_definitions[exp_name]["away"]]
    df_exp["Total"] = df_exp["Home"] + df_exp["Away"]

    for side in ["Home", "Away", "Total"]:
        df_exp[f"Base_{side}"] = df_exp.groupby(event_col)[side].transform("first")
        df_exp[f"Delta_{side}"] = df_exp.groupby(event_col)[side].diff().fillna(0)

    df_exp["Fav"] = np.where(df_exp["Favourite"] == "Home", df_exp["Home"], df_exp["Away"])
    df_exp["Dog"] = np.where(df_exp["Favourite"] == "Home", df_exp["Away"], df_exp["Home"])

    for side in ["Fav", "Dog"]:
        df_exp[f"Base_{side}"] = df_exp.groupby(event_col)[side].transform("first")
        df_exp[f"Delta_{side}"] = df_exp.groupby(event_col)[side].diff().fillna(0)

    df_exp["Change_Total"] = df_exp["Total"] - df_exp["Base_Total"]
    df_exp["Change_Fav"] = df_exp["Fav"] - df_exp["Base_Fav"]
    df_exp["Change_Dog"] = df_exp["Dog"] - df_exp["Base_Dog"]

    change_col = f"Change_{view_type}"
    delta_col = f"Delta_{view_type}"
    df_view = df_exp[df_exp[delta_col] != 0].copy()
    if df_view.empty:
        st.warning(f"No data to display for {selection} after applying deltas and filters.")
        continue

    df_view["EXP_Change"] = df_view[change_col]
    avg_exp_change = df_view.groupby("Time Band", observed=False)["EXP_Change"].mean()
    match_counts = df_view.groupby("Time Band", observed=False)[event_col].nunique()
    all_bands = df["Time Band"].cat.categories
    avg_exp_change = avg_exp_change.reindex(all_bands, fill_value=np.nan)
    match_counts = match_counts.reindex(all_bands, fill_value=0)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(avg_exp_change.index, avg_exp_change.values, color='black', marker='o')
    ax1.set_ylabel("Avg EXP Change")
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.set_xlabel("Time Band")
    ax1.set_xticks(range(len(avg_exp_change.index)))
    ax1.set_xticklabels(avg_exp_change.index, rotation=45)
    y_min, y_max = avg_exp_change.min(skipna=True), avg_exp_change.max(skipna=True)
    margin = 0.1 * max(abs(y_min), abs(y_max), 0.1)
    ax1.set_ylim(y_min - margin, y_max + margin)

    ax2 = ax1.twinx()
    ax2.bar(match_counts.index, match_counts.values, width=0.6, color='skyblue', alpha=0.4)
    ax2.set_ylabel("Number of Unique Matches", color="skyblue")

    plt.title(f"{exp_name} – {view_type} Change")
    plt.grid(True, axis='y')
    st.pyplot(fig)
