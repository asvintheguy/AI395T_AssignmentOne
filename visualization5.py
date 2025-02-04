import pandas as pd
import plotly.express as px
import numpy as np

# Load relevant tables
icustays_df = pd.read_csv("../mimic-iii-clinical-database-1.4/ICUSTAYS.csv")
services_df = pd.read_csv("../mimic-iii-clinical-database-1.4/SERVICES.csv")
procedures_df = pd.read_csv("../mimic-iii-clinical-database-1.4/PROCEDUREEVENTS_MV.csv")
d_items_df = pd.read_csv("../mimic-iii-clinical-database-1.4/D_ITEMS.csv")
input_df = pd.read_csv("../mimic-iii-clinical-database-1.4/INPUTEVENTS_MV.csv")


# Convert datetime columns to datetime objects
icustays_df["INTIME"] = pd.to_datetime(icustays_df["INTIME"])
icustays_df["OUTTIME"] = pd.to_datetime(icustays_df["OUTTIME"])

# Calculate ICU length of stay in hours
icustays_df["LOS_HOURS"] = (
    icustays_df["OUTTIME"] - icustays_df["INTIME"]
).dt.total_seconds() / 3600

# ICU Resource Usage
icu_resources = (
    icustays_df.groupby("FIRST_CAREUNIT")
    .agg({"SUBJECT_ID": "count", "LOS_HOURS": ["sum", "mean"]})
    .round(2)
)
icu_resources.columns = ["PATIENT_COUNT", "TOTAL_HOURS", "AVG_LOS"]
icu_resources = icu_resources.reset_index()

# Service Resource Usage
service_resources = (
    services_df.groupby("CURR_SERVICE").size().reset_index(name="SERVICE_COUNT")
)

# Procedure and Input Categories
d_items_df["CATEGORY"] = d_items_df["CATEGORY"].fillna("Other")

# Initialize rows
resource_rows = []

# Add ICU data
for _, row in icu_resources.iterrows():
    if pd.notna(row["FIRST_CAREUNIT"]):
        resource_rows.append(
            {
                "RESOURCE_TYPE": "ICU Care",
                "CATEGORY": str(row["FIRST_CAREUNIT"]),
                "METRIC": "Patient Days",
                "VALUE": float(row["TOTAL_HOURS"]) / 24,  # Convert to days
                "COUNT": int(row["PATIENT_COUNT"]),
            }
        )

# Add Service data
for _, row in service_resources.iterrows():
    if pd.notna(row["CURR_SERVICE"]):
        resource_rows.append(
            {
                "RESOURCE_TYPE": "Medical Services",
                "CATEGORY": str(row["CURR_SERVICE"]),
                "METRIC": "Service Episodes",
                "VALUE": float(row["SERVICE_COUNT"]),
                "COUNT": int(row["SERVICE_COUNT"]),
            }
        )

# Add Procedure data by category
proc_categories = (
    procedures_df.merge(d_items_df[["ITEMID", "CATEGORY"]], on="ITEMID", how="left")
    .groupby("CATEGORY")
    .agg({"ITEMID": "count", "VALUE": "sum"})
    .reset_index()
)

for _, row in proc_categories.iterrows():
    if pd.notna(row["CATEGORY"]):
        resource_rows.append(
            {
                "RESOURCE_TYPE": "Procedures",
                "CATEGORY": str(row["CATEGORY"]),
                "METRIC": "Procedure Count",
                "VALUE": float(
                    row["VALUE"] if pd.notna(row["VALUE"]) else row["ITEMID"]
                ),
                "COUNT": int(row["ITEMID"]),
            }
        )

# Add Input data by category
input_categories = (
    input_df.merge(d_items_df[["ITEMID", "CATEGORY"]], on="ITEMID", how="left")
    .groupby("CATEGORY")
    .agg({"ITEMID": "count", "AMOUNT": "sum"})
    .reset_index()
)

for _, row in input_categories.iterrows():
    if pd.notna(row["CATEGORY"]):
        resource_rows.append(
            {
                "RESOURCE_TYPE": "Medical Inputs",
                "CATEGORY": str(row["CATEGORY"]),
                "METRIC": "Input Amount",
                "VALUE": float(
                    row["AMOUNT"] if pd.notna(row["AMOUNT"]) else row["ITEMID"]
                ),
                "COUNT": int(row["ITEMID"]),
            }
        )

# Create resource dataframe
resource_df = pd.DataFrame(resource_rows).dropna()

# Create the sunburst diagram
fig = px.sunburst(
    resource_df,
    path=["RESOURCE_TYPE", "CATEGORY", "METRIC"],
    values="VALUE",
    color="COUNT",
    title="Hospital Resource Utilization Pattern",
    width=1000,
    height=800,
    color_continuous_scale="Viridis",
)

# Update layout
fig.update_layout(
    title_x=0.5, title_font_size=20, coloraxis_colorbar_title="Usage Count"
)

# Save the visualization
fig.write_html("resource_utilization_sunburst.html")