import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the required tables with proper settings
admissions_df = pd.read_csv("../mimic-iii-clinical-database-1.4/ADMISSIONS.csv")
patients_df = pd.read_csv("../mimic-iii-clinical-database-1.4/PATIENTS.csv")

# Print column names to verify structure
print("Admissions columns:", admissions_df.columns.tolist())
print("Patients columns:", patients_df.columns.tolist())

# Check the insurance types
insurance_dist = admissions_df["INSURANCE"].value_counts()
print("Insurance types distribution:", insurance_dist)

# Coimbine the admissions and patients dataframes
merged_df = admissions_df.merge(patients_df[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID")

# Calculate length of stay and age
merged_df["LOS"] = (
    pd.to_datetime(merged_df["DISCHTIME"]) - pd.to_datetime(merged_df["ADMITTIME"])
).dt.total_seconds() / (24 * 60 * 60)

merged_df["ADMITTIME"] = pd.to_datetime(merged_df["ADMITTIME"])
merged_df["DOB"] = pd.to_datetime(merged_df["DOB"])
merged_df["AGE"] = merged_df["ADMITTIME"].dt.year - merged_df["DOB"].dt.year

# Create the different age groups
merged_df["AGE_GROUP"] = pd.cut(
    merged_df["AGE"],
    bins=[0, 30, 50, 70, 100],
    labels=["0-30", "31-50", "51-70", "70+"],
)

# Create visualization
plt.figure(figsize=(15, 8))

# Set color palette for age groups
colors = {"0-30": "#2ecc71", "31-50": "#3498db", "51-70": "#9b59b6", "70+": "#e74c3c"}

# Create the box plot for each of the groups and insurances
sns.boxplot(
    data=merged_df[merged_df["LOS"] <= merged_df["LOS"].quantile(0.95)],
    x="INSURANCE",
    y="LOS",
    hue="AGE_GROUP",
    palette=colors,
    width=0.8,
    linewidth=1.5,
    showfliers=False,
)


# Addd title and labels
plt.title(
    "Length of Stay Distribution by Insurance Type and Age Group\n(95th percentile of LOS shown)",
    pad=20,
    fontsize=12,
)
plt.xlabel("Insurance Type", fontsize=11)
plt.ylabel("Length of Stay (days)", fontsize=11)

# Rotate x-axis labels to look betteer
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)

# Add a legend for the age groups
plt.legend(
    title="Age Group",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    title_fontsize=11,
    fontsize=10,
)

# Save the plot
plt.tight_layout()
plt.savefig("insurance_los.png")