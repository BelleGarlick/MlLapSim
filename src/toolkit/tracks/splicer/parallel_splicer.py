import multiprocessing
from typing import List, Optional, Union
from multiprocessing import cpu_count

from .models import SplicerInput
from .splicer import splice
from toolkit.tracks.splicer.models.parallel_splicer_models import (
    ParallelSplicerInput,
    ParallelSplicerOutput
)
from toolkit.tracks.models import Track

"""This module allows for parallel splicing of data utilising the multi-
processing library. `parallel_splicer` function takes a list of splicer
inputs with some other optional params which speeds up the splicing process.
"""


def __splicer_wrapper(parallel_input: ParallelSplicerInput) -> Union[str, None, Track]:
    """This function wraps the input to the multiprocess splicer function
    allowing us to determine whether to return the output or input if there
    was an error.

    Usage:
        outputs = map(__splicer_wrapper, inputs)

    Args:
        parallel_input: The parallel splicer input which contains the input
            to be fed to the splicer function.

    Returns:
        None: If return outputs was false
        The input: If there was an error
        The output: If it was succesfull
    """
    try:
        result = splice(parallel_input.splicer_input)
        if parallel_input.return_output:
            return result
        else:
            return None
    except Exception as e:
        return str(e)


def parallel_splice(
        inputs: List[SplicerInput],
        batch_size: Optional[int] = None,
        cores: Optional[int] = None,
        return_output=True
) -> ParallelSplicerOutput:
    """Run the splicer function in parallel across CPU cores.

    This function splits the inputs into a series of batches which are spread
    over the cpu cores. This means, for loading data in bulk, you can obtain a
    massive reduction in compute time to splice a dataset of tracks.

    Args:
        inputs: List of inputs to feed to the splicer function
        batch_size: Number of items in each batch. Larger batch size results in
            fast compute time but less logging since logs are only printed at
            the start of a batch. This value will default to cores * 8.
            Additionally, the batch size given should be a multiple of the
            number of cores used.
        cores: The number of cores to compute across, if None is given then the
            number of cores used will be equal to the cpu count of the cpu.
        return_output: If False, the output of the splicing will not be
            given. This is for memory constrained deviced whereby the tracks
            are written to disk and therefor not needed to be returned in
            memory.

    Returns:
        A breakdown of failed tracks and (if return_output is not false) the
        spliced tracks.
    """
    # batch size 6 with 2 cores means 3 items per cores
    if cores is None:
        cores = cpu_count()
    if batch_size is None:
        batch_size = cores * 8

    print(f"Parellalising on {cores} cores with batch size {batch_size} over {len(inputs)} items.")

    # Re-model the inputs to use the parallel splicer input which contains
    # other instructions for paralell splicings
    inputs: List[ParallelSplicerInput] = [
        ParallelSplicerInput(
            splicer_input=x,
            return_output=return_output
        )
        for x in inputs
    ]

    # Split items into batches
    batches: List[List[ParallelSplicerInput]] = []
    for i in range(0, len(inputs), batch_size):
        batches.append(inputs[i:i + batch_size])
    print(f"{len(batches)} batches.")

    # Multi-process the results
    results = []
    with multiprocessing.Pool(cores) as p:
        for i, batch in enumerate(batches):
            # multiprocess the encoding
            print(f"\r{i}/{len(batches)}", end="")
            results += p.map(__splicer_wrapper, batch)
    print(f"\r{len(results)} items complete.")

    # Filter the response type to see what the response type is
    # which will be returned
    valid_results: List[Track] = []
    error_results: List[str] = []
    for result in results:
        if isinstance(result, Track):
            valid_results.append(result)
        elif isinstance(result, str):
            error_results.append(result)

    print(f"{len(error_results)} failed.")
    if error_results:
        print(f"First Error: {error_results[0]}")

    return ParallelSplicerOutput(
        spliced=valid_results,
        errors=error_results
    )
