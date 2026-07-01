import numpy as np
from typing import List, Any

def at_indexes(items: List[Any], indexes: List[int]) -> List[Any]:
    """Get items at given indexes"""
    arr = np.asarray(items, dtype=object)
    return arr[indexes].tolist()
