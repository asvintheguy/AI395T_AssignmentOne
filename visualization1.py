import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataframes
prescriptions_df = pd.read_csv("../mimic-iii-clinical-database-1.4/PRESCRIPTIONS.csv")
admissions_df = pd.read_csv("../mimic-iii-clinical-database-1.4/ADMISSIONS.csv")
patients_df = pd.read_csv("../mimic-iii-clinical-database-1.4/PATIENTS.csv")

# Print column names to verify structure
print("Prescriptions columns:", prescriptions_df.columns.tolist())
print("Admissions columns:", admissions_df.columns.tolist())
print("Patients columns:", patients_df.columns.tolist())

# Get top 5 medications
top_meds = prescriptions_df["DRUG"].value_counts().head(5)
top_med_list = top_meds.index.tolist()

print("Top 5 medications:")
print(top_meds)

# Filter prescriptions to include only top 5 medications
prescriptions_filtered = prescriptions_df[prescriptions_df["DRUG"].isin(top_med_list)]

# Merge the dataframes one by one
merged_df = prescriptions_filtered.merge(
    admissions_df[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"]],
    on=["SUBJECT_ID", "HADM_ID"],
)

# Calculate length of stay (LOS)
merged_df["LOS"] = (
    pd.to_datetime(merged_df["DISCHTIME"]) - pd.to_datetime(merged_df["ADMITTIME"])
).dt.total_seconds() / (24 * 60 * 60)

# Create the visualization
plt.figure(figsize=(12, 8))

# Create violin plot
sns.violinplot(
    data=merged_df,
    x="DRUG",
    y="LOS",
    inner="box",
    cut=0,
    density_norm="width",
)

# Labels
plt.xticks(rotation=45, ha="right")
plt.xlabel("Medication")
plt.ylabel("Length of Stay (days)")
plt.title("Distribution of Length of Stay by Medication")

# Save the plot
plt.tight_layout()
plt.savefig("medication_los.png")