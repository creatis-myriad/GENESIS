import csv

import rootutils

from genesis.data.utils.io import load_and_clean_tabular_features

data = load_and_clean_tabular_features(
    rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/global_features.csv"
)

# Compute binary targets from available data
binary_targets = {
    "nt-probnp_elevated": (elevated_nt_probnp := data["nt-probnp"] >= 600),
    "troponin_elevated": (elevated_troponin := data["troponin"] > 14),
    "enzymes_elevated": elevated_nt_probnp | elevated_troponin,
    "risk_treated_elevated": data["risk_treated"] >= 2,
    "risk_ESC-2014_elevated": data["risk_ESC-2014"] >= 2,
    "risk_ESC-2019_elevated": data["risk_ESC-2019"] >= 2,
}
# Make sure the binary targets are integers (0 and 1) rather than booleans (False and True)
binary_targets = {k: v.astype(int) for k, v in binary_targets.items()}

# Save the new binary targets in the clinical data DataFrame and CSV
data = data.assign(**binary_targets)
data.to_csv(
    rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/global_features_processed.csv",
    index=True,
    quoting=csv.QUOTE_NONNUMERIC,
)
