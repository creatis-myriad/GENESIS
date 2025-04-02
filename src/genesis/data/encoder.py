import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy data types."""

    def default(self, obj: object) -> object:
        """Convert NumPy data types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(self).default(obj)
