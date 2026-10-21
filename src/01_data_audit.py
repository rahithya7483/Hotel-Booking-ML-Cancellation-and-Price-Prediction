"""Step 1 - Data audit: what is wrong with the raw data and what did cleaning change?"""
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS
from prep import load_raw, get_clean_data
from utils import savefig

raw = load_raw()
clean = get_clean_data()

# ---- before / after table --------------------------------------------------
audit = pd.DataFrame({
    "item": [
        "rows", "columns", "exact duplicate rows", "missing cells (children)",
        "missing cells (country)", "missing cells (agent)", "missing cells (company)",
        "bookings with zero guests", "rows with adr <= 0 (regression only)",
        "cancellation rate",
    ],
    "raw_data": [
        len(raw), raw.shape[1], int(raw.duplicated().sum()),
        int(raw["children"].isna().sum()), int(raw["country"].isna().sum()),
        int(raw["agent"].isna().sum()), int(raw["company"].isna().sum()),
        int(((raw["adults"] + raw["children"].fillna(0) + raw["babies"]) == 0).sum()),
        int((raw["adr"] <= 0).sum()),
        round(raw["is_canceled"].mean(), 4),
    ],
    "after_cleaning": [
        len(clean), clean.shape[1], int(clean.duplicated().sum()),
        int(clean["children"].isna().sum()), int(clean["country"].isna().sum()),
        "converted to has_agent", "converted to has_company",
        int(((clean["adults"] + clean["children"] + clean["babies"]) == 0).sum()),
        int((clean["adr"] <= 0).sum()),
        round(clean["is_canceled"].mean(), 4),
    ],
})
audit.to_csv(RESULTS / "data_audit.csv", index=False)
print(audit.to_string(index=False))

# ---- figures ---------------------------------------------------------------
# 1. class balance
fig, ax = plt.subplots(figsize=(4.5, 3.5))
counts = clean["is_canceled"].value_counts().sort_index()
ax.bar(["Not cancelled", "Cancelled"], counts.values, color=["#4c72b0", "#c44e52"])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\n({v / len(clean):.1%})", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, counts.max() * 1.2)
ax.set_title("Class balance (after removing duplicates)")
savefig("01_class_balance.png")

# 2. cancellation rate by month and by deposit type
order = ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
m = clean.groupby("arrival_date_month")["is_canceled"].mean().reindex(order)
axes[0].bar(range(12), m.values, color="#4c72b0")
axes[0].set_xticks(range(12))
axes[0].set_xticklabels([x[:3] for x in order])
axes[0].set_title("Cancellation rate by arrival month")
d = clean.groupby("deposit_type")["is_canceled"].mean()
axes[1].bar(d.index, d.values, color="#55a868")
axes[1].set_title("Cancellation rate by deposit type")
savefig("02_cancel_rate_by_month_deposit.png")

# 3. price (adr) distribution - shows skew and outliers
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(clean["adr"].clip(-10, 500), bins=60, color="#8172b2")
axes[0].set_title("ADR (price per night) - clipped at 500")
axes[1].hist(clean.loc[clean["adr"].between(0.01, 500), "adr"], bins=60, color="#8172b2")
axes[1].set_title("ADR after removing adr<=0 and adr>500")
savefig("03_adr_distribution.png")

print("\nSaved results/data_audit.csv and 3 figures.")
