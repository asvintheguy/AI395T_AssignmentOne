import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Check data
print("Loading data...")
labevents = pd.read_csv("../mimic-iii-clinical-database-1.4/LABEVENTS.csv")
d_labitems = pd.read_csv("../mimic-iii-clinical-database-1.4/D_LABITEMS.csv")
admissions = pd.read_csv("../mimic-iii-clinical-database-1.4/ADMISSIONS.csv")

print("Labevents columns:", labevents.columns.tolist())
print("D_labitems columns:", d_labitems.columns.tolist())
print("Admissions columns:", admissions.columns.tolist())

# Print labevents flag
print("Labevents flag:", labevents["FLAG"])


# Get the top 10 most frequent lab tests
print("Getting top 10 most frequent lab tests...")
lab_counts = labevents["ITEMID"].value_counts()
top_labs = d_labitems[d_labitems["ITEMID"].isin(lab_counts.index[:10])]
lab_ids = top_labs["ITEMID"].tolist()
for _, lab in top_labs.iterrows():
    count = lab_counts[lab["ITEMID"]]
    print(f"{lab['LABEL']}: {count:,} measurements")

# Get lab values for the selected tests
print("Getting lab values for the selected tests...")
lab_values = labevents[labevents["ITEMID"].isin(lab_ids)].copy()
lab_values = lab_values.merge(d_labitems[["ITEMID", "LABEL"]], on="ITEMID", how="left")
lab_values = lab_values.merge(
    admissions[["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]], on="HADM_ID", how="left"
)

# Calculate abnormalities based on lab FLAGS...
print("Calculating abnormalities based on lab FLAGS...")
abnormal_stats = {}
labs_list = []
for lab_id in lab_ids:
    lab_data = lab_values[lab_values["ITEMID"] == lab_id]
    lab_name = top_labs[top_labs["ITEMID"] == lab_id]["LABEL"].iloc[0]

    # Remove null values for VALUENUM
    lab_data = lab_data[lab_data["VALUENUM"].notna()]

    # Identify abnormal values using FLAG column
    abnormal = lab_data[lab_data["FLAG"].notna()]

    # Calculate mortality rate for abnormal values
    mortality_rate = (abnormal["HOSPITAL_EXPIRE_FLAG"] == 1).mean() * 100

    # Store results
    abnormal_stats[lab_name] = {
        "mortality_rate": mortality_rate,
        "n_abnormal": len(abnormal),
        "n_total": len(lab_data),
        "percent_abnormal": (len(abnormal) / len(lab_data)) * 100,
    }
    labs_list.append(lab_name)

# Create a matrix for the heatmap
print("Creating a matrix for the heatmap...")
n_labs = len(labs_list)
matrix = np.zeros((n_labs, n_labs))


# Calculate mortality rates for combinations
print("Calculating mortality rates for combinations...")
for i, lab1 in enumerate(labs_list):
    for j, lab2 in enumerate(labs_list):
        if i == j:
            matrix[i, j] = abnormal_stats[lab1]["mortality_rate"]
            continue

        lab1_data = lab_values[lab_values["LABEL"] == lab1]
        lab2_data = lab_values[lab_values["LABEL"] == lab2]

        # Get common hospital admissions with abnormal values in both tests
        abnormal1_hadm = lab1_data[lab1_data["FLAG"].notna()]["HADM_ID"]
        abnormal2_hadm = lab2_data[lab2_data["FLAG"].notna()]["HADM_ID"]

        common_hadm = set(abnormal1_hadm) & set(abnormal2_hadm)

        if len(common_hadm) == 0:
            matrix[i, j] = 0
            continue

        # Calculate mortality rate for common admissions
        mortality = (
            lab_values[lab_values["HADM_ID"].isin(common_hadm)][
                "HOSPITAL_EXPIRE_FLAG"
            ].mean()
            * 100
        )

        matrix[i, j] = mortality

# Create the heatmap
print("Creating the heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(
    matrix,
    xticklabels=labs_list,
    yticklabels=labs_list,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    vmin=0,
    vmax=np.max(matrix),
)

# Labels
plt.title(
    "Mortality Rates (%) for Abnormal Lab Value Combinations\n(Based on Laboratory FLAG Values)"
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

# Save visualization
plt.tight_layout()
plt.savefig("lab_combinations_mortality.png")
