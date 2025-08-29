from enum import IntEnum


class ArteryLevel(IntEnum):
    """Enum of hierarchical levels of arteries considered by pulmonary embolism obstruction scores."""

    ROOT = 1
    MEDIASTINAL = 2
    LOBAR = 3
    SEGMENTAL = 4
