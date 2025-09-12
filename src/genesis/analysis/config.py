from enum import IntEnum


class ArteryLevel(IntEnum):
    """Enum of hierarchical levels of arteries considered by pulmonary embolism obstruction scores."""

    ROOT = 1
    MEDIASTINAL = 2
    LOBAR = 3
    SEGMENTAL = 4


PERSEVERE_ATTRS_LABELS = {
    "mastora_central": "Mastora (central) score",
    "mastora_peripheral": "Mastora (peripheral) score",
    "mastora_global": "Mastora (global) score",
    "qanadli": "Qanadli score",
    "risk_treated": "PE risk based on which patients were treated",
    "risk_treated_elevated": "PE risk based on which patients were treated >= 2",
    "risk_ESC-2014": "PE risk based on ESC 2014 Guidelines",
    "risk_ESC-2014_elevated": "PE risk based on ESC 2014 Guidelines >= 2",
    "risk_ESC-2019": "PE risk based on ESC 2019 Guidelines",
    "risk_ESC-2019_elevated": "PE risk based on ESC 2019 Guidelines >= 2",
    "spesi": "SPESI Score",
    "nt-probnp": "NT-proBNP",
    "nt-probnp_elevated": "NT-proBNP > 600",
    "troponin": "Troponin",
    "troponin_elevated": "Troponin > 14",
    "age": "Age (years)",
    "history_cancer": "History of cancer",
    "history_cpd": "History of chronic cardiopulmonary disease",
    "heart_rate": "Heart rate (bpm)",
    "systolic_bp": "Systolic blood pressure (mmHg)",
    "spO2": "Oxygen saturation (%)",
    "inverted_rv-lv_ratio": "Inverted RV/LV ratio",
}
PERSEVERE_ATTRS_CATEGORIES = {
    "risk_treated": [0, 1, 2],
    "risk_ESC-2014": [0, 1, 2],
    "risk_ESC-2019": [0, 1, 2],
    "spesi": [0, 1, 2],
    "history_cancer": [0, 1],
    "history_cpd": [0, 1],
    "inverted_rv-lv_ratio": [0, 1],
    **{attr: [0, 1] for attr in PERSEVERE_ATTRS_LABELS if attr.endswith("_elevated")},
}
PERSEVERE_ATTRS_RANGES = {
    "mastora_central": (0.0, 1.0),
    "mastora_peripheral": (0.0, 1.0),
    "mastora_global": (0.0, 1.0),
    "qanadli": (0.0, 1.0),
}
PERSEVERE_ATTRS_SCALES = {
    "nt-probnp": "log",
    "troponin": "log",
}
PERSEVERE_GRAPH_SCORES = [
    "mastora_central",
    "mastora_peripheral",
    "mastora_global",
    "qanadli",
]
