cimport cython
from libc.math cimport (
    pow,
    sqrt
)


@cython.wraparound(False)
cpdef at_indexes(items, list[int] indexes):
    cdef int count = len(indexes)
    cdef int i = 0

    cdef objects = [0] * count

    for i in range(count):
        objects[i] = items[indexes[i]]

    return objects


@cython.wraparound(False)
@cython.cdivision(True)
cpdef list roll(list items, int shift):
    cdef int count = len(items)
    if count == 0 or shift == 0: return items

    cdef int i = 0
    cdef rolled_items = [0] * count

    cdef int shifted_index = shift % count

    for i in range(count):
        rolled_items[shifted_index] = items[i]

        shifted_index += 1
        if shifted_index == count:
            shifted_index = 0

    return rolled_items
