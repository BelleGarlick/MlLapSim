import numpy as np
from typing import List, Any

def at_indexes(items: List[Any], indexes: List[int]) -> List[Any]:
    """Get items at given indexes"""
    arr = np.asarray(items, dtype=object)
    return arr[indexes].tolist()

def roll(items: List[Any], amount: int) -> List[Any]:
    """Roll items by given amount"""
    arr = np.asarray(items, dtype=object)
    return np.roll(arr, amount).tolist()
