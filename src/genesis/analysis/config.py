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
    "risk": "ESC Guidelines risk level",
    "spesi": "SPESI Score",
    "bnp": "BNP",
    "troponin": "Troponin",
    "age": "Age (years)",
    "cancer_history": "History of cancer",
    "cpd_history": "History of chronic cardiopulmonary disease",
    "heart_rate": "Heart rate (bpm)",
    "systolic_bp": "Systolic blood pressure (mmHg)",
    "spO2": "Oxygen saturation (%)",
    "inverted_rv-lv_ratio": "Inverted RV/LV ratio",
}
PERSEVERE_ATTRS_CATEGORIES = {
    "risk": [0, 1, 2],
    "spesi": [0, 1, 2],
    "cancer_history": [0, 1],
    "cpd_history": [0, 1],
    "inverted_rv-lv_ratio": [0, 1],
}
