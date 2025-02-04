import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

transfers = pd.read_csv("../mimic-iii-clinical-database-1.4/TRANSFERS.csv")
transfers["INTIME"] = pd.to_datetime(transfers["INTIME"])
transfers["OUTTIME"] = pd.to_datetime(transfers["OUTTIME"])
transfers["IN_HOUR"] = transfers["INTIME"].dt.hour
transfers["OUT_HOUR"] = transfers["OUTTIME"].dt.hour

# Categorize ICU units
icu_units = {
    "MICU": ("Medical ICU", "#FF9999"),
    "CCU": ("Cardiac Care Unit", "#66B2FF"),
    "CSRU": ("Cardiac Surgery Recovery", "#99FF99"),
    "SICU": ("Surgical ICU", "#FFCC99"),
    "TSICU": ("Trauma/Surgical ICU", "#FF99FF"),
    "NICU": ("Neuro ICU", "#99FFFF"),
}

# Filter for ICU transfers and add unit descriptions
icu_transfers = transfers[transfers["CURR_CAREUNIT"].isin(icu_units.keys())].copy()
icu_transfers["UNIT_DESC"] = icu_transfers["CURR_CAREUNIT"].map(
    lambda x: icu_units[x][0]
)

# Create figure
plt.figure(figsize=(24, 8))

# Set up bar positions variables
hours = np.arange(24)
bar_width = 0.4
bar_gap = 0.3
group_gap = 0.6

# Calculate positions for bars
r1 = np.arange(len(hours)) * (
    2 * bar_width + bar_gap + group_gap
)
r2 = r1 + bar_width + bar_gap

# Plot transfers for each unit
bottom_in = np.zeros(24)
bottom_out = np.zeros(24)

for unit, (desc, color) in icu_units.items():
    unit_data = icu_transfers[icu_transfers["CURR_CAREUNIT"] == unit]
    
    # Get counts
    in_counts = unit_data["IN_HOUR"].value_counts().reindex(hours).fillna(0)
    out_counts = unit_data["OUT_HOUR"].value_counts().reindex(hours).fillna(0)

    # Plot stacked bars
    plt.bar(r1, in_counts, bar_width, bottom=bottom_in, label=desc, color=color)  
    plt.bar(r2, out_counts, bar_width, bottom=bottom_out, color=color, hatch="...") # Add a /// to clearly show the out bars

    bottom_in += in_counts
    bottom_out += out_counts

# Add labels
plt.xlabel("Hour of Day", labelpad=20)
plt.ylabel("Number of Transfers")
plt.title("ICU Transfer Patterns Throughout the Day", pad=20, fontsize=14)

for i in range(len(hours)):
    # Add IN/OUT labels
    plt.text(r1[i], -300, "IN", ha="center", va="bottom", fontweight="bold")
    plt.text(r2[i], -300, "OUT", ha="center", va="bottom", fontweight="bold")

    # Add hour label below, centered between IN and OUT
    hour_center = r1[i] + (bar_width + bar_gap) / 2
    plt.text(hour_center, -400, f"{i:02d}:00", ha="center", va="top", rotation=0)

    # Separate hours with vertical lines
    if i < len(hours) - 1:
        sep_x = (r2[i] + r1[i + 1]) / 2
        plt.axvline(x=sep_x, color="gray", linestyle=":", alpha=0.3)

# Set x-axis ticks and limits
plt.xticks([])
plt.xlim(r1[0] - bar_width, r2[-1] + bar_width)

plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="ICU Units")

# Adjust layout and save
plt.tight_layout()
plt.savefig("icu_transfer_patterns.png")
