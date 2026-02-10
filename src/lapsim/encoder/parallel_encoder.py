import multiprocessing
import random
import time
from typing import List, Optional, Callable

from lapsim.encoder import encode
from lapsim.encoder.encoder_input import EncoderInput
from lapsim.encoder.partition import Partition


"""This module handles parallel encoding tracks. Building on the encoder 
functionality, this file implements the multiprocessing to allow for bulk 
encoding at speed and combining multiple partitions together.
"""


def parallel_encode(
        inputs: List[EncoderInput],
        n_partitions: int,
        batch_size: int,
        cores: int,
        path: Optional[Callable[[int], str]] = None,
        return_partitions: bool = True
) -> Optional[List[Partition]]:
    """Encode the given inputs.

    The random library is used to randomise the partitions inputs go to. So in
    order to seed the encoding, use random.seed. Also, tracks will be flipped
    on a case-by-case basis. So, if you want to both flip a track and have a
    regular track, you need to pass the same input twice except one is flipped
    and one is not.

    Args:
        inputs: List of inputs each of which is passed to the encode function.
        n_partitions: Number of partitions to output. If this number is None
            or less than 1, then each track will be exported to individual
            partitions.
        batch_size: How many items to process befoe logging an update.
            Higher batch size should be faster since the process will be stopped
            fewer times. This should be a multiple of the number of cores used.
        cores: Number of cores to utilise to run the encoder.
        path: By default, partitions are not saved, if a path is given then the
            partition will be saved to the given path. The path is a callback
            function which gives the current index of the partition. A simple
            argument for this parameter would be something like:
                `lambda idx: f"partition-{idx}.json"`.
        return_partitions: If true (default), then all partitions will be
            returned after being encoded. For encoding large number of tracks,
            this value should be set to false. As it may lead to all memory
            being used up.

    Returns:
        A list of partitions if `return_partitions` is true, otherwise this
        function will return None
    """
    all_partitions = []

    if n_partitions is None:
        n_partitions = 0
    # batch size 6 with 2 cores means 3 items per cores
    if cores is None:
        cores = multiprocessing.cpu_count()
    if batch_size is None:
        batch_size = cores * 8

    # Check whether to compute individually or not
    if n_partitions < 1:
        # If n_partitions < 1 then each track should be encoded individually
        partitions = __encode_items(inputs, batch_size=batch_size, cores=cores)

        # Each item in the partition list should be exported individually
        for i, partition in enumerate(partitions):
            if path: partition.save(path(i))
            if return_partitions: all_partitions.append(partition)

    else:
        # If n_partitions >= 1 then group all inputs into partition groups, encode each item
        # then combine into one bigger partition per group.
        for i, partition_items in enumerate(create_partition_groups(inputs, n_partitions=n_partitions)):
            partitions = __encode_items(partition_items, batch_size=batch_size, cores=cores)
            partition = Partition.combine(partitions)

            if path:
                partition.save(path(i))
            if return_partitions: all_partitions.append(partition)

    if return_partitions:
        return all_partitions

    return None


def __encode_items(items: List[EncoderInput], batch_size: int, cores: int) -> List[Partition]:
    """Encode each of the given list of inputs into individual partitions which
    can later be combined into one larger partitions. This function will encode
    each track using multiprocesing to utilise all cpu cores to speed up the
    process.

    Args:
        items: List if inputs to feed to the encoder
        batch_size: How many items to process befoe logging an update.
            Higher batch size should be faster since the process will be stopped
            fewer times. This should be a multiple of the number of cores used.
        cores: How many threads to run the encoder over.

    Returns:
        List of partitions mapping to each of the input tracks
    """
    print(f"\tLoading {len(items)} tracks.")
    start = time.time()

    # Init multiprocessing, to speed up data encoding
    print("\rLoaded 0 items.", end="")

    if cores is None:
        cores = multiprocessing.cpu_count()

    # Split items into batches
    batches: List[List[EncoderInput]] = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    print(f"{len(batches)} batches.")

    track_partitions: List[Partition] = []

    with multiprocessing.Pool(cores) as p:
        for i, batch in enumerate(batches):
            # multiprocess the encoding
            print(f"\r{i}/{len(batches)}", end="")
            track_partitions += p.map(encode, batch)

    print(f"\rData Loaded. {len(items)} tracks in {time.time() - start}s.")

    return track_partitions


def create_partition_groups(inputs: List[EncoderInput], n_partitions: int) -> List[List[EncoderInput]]:
    """This function will split the given list of inputs and group them into
    the number of partitions given. The minimum number of partitions is 1.

    Args:
        inputs: List of inputs that will be passed to the encoder
        n_partitions: The number of partitions to load

    Returns:
        List of groups of lists of inputs
    """
    if n_partitions < 1:
        n_partitions = 1

    items = [item for item in inputs]
    random.shuffle(items)

    # Create the initial empty partitions to add items to
    partitioned_tracks = [[] for _ in range(n_partitions)]

    # Iterate through all items and add it to the correct partition
    for index, item in enumerate(items):
        partitioned_tracks[index % n_partitions].append(item)

    return partitioned_tracks
