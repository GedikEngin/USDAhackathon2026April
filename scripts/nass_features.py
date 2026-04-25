import pandas as pd

df = pd.read_csv("nass_corn_5states_2005_2024.csv")

# Compute irrigation share where data exists (CO and NE).
# For other states, irr cols are NaN; treat as rainfed.
df["irrigated_share"] = (
    df["acres_harvested_irr"] / df["acres_harvested_all"]
).fillna(0).clip(0, 1)

# Derive non-irrigated acres for CO and NE
df["acres_harvested_noirr_derived"] = (
    df["acres_harvested_all"] - df["acres_harvested_irr"].fillna(0)
)

# Target
df["yield_target"] = df["yield_bu_acre_all"]

# Harvest ratio (planted -> harvested), proxy for in-season abandonment
df["harvest_ratio"] = (
    df["acres_harvested_all"] / df["acres_planted_all"]
).clip(0, 1)

# Final feature columns
out = df[[
    "GEOID", "year", "state_alpha", "county_name",
    "yield_target", "irrigated_share", "harvest_ratio",
    "acres_harvested_all", "acres_planted_all",
    "yield_bu_acre_irr",  # CO/NE only, sparse but useful
]].dropna(subset=["yield_target"])

print(f"Rows: {len(out):,}")
print(f"\nIrrigation share by state:")
print(out.groupby("state_alpha")["irrigated_share"].agg(["mean", "min", "max"]))

out.to_csv("nass_corn_5states_features.csv", index=False)
print(f"\nSaved to nass_corn_5states_features.csv")