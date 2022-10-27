import numpy as np

from contextlib import contextmanager
from time import time as t
from typing import Union


@contextmanager
def timer_context(argument: str,
                  level: Union[str, None] = None,
                  unit: Union[str, None] = None,
                  verbose: bool = True
                  ):
    """Context manager to measure time execution."""
    if verbose:
        t0 = t()
        yield
        t1 = t()
        if unit == "hour":
            time_execution = np.round((t1 - t0) / (3600), 2)
        elif unit == "minute":
            time_execution = np.round((t1 - t0) / 60, 2)
        elif unit == "second":
            time_execution = np.round((t1 - t0), 2)
        print(f"{level}Time to calculate {argument}: {time_execution} {unit}s")
    else:
        yield


@contextmanager
def creation_context(argument: str,
                     level: Union[str, None] = None,
                     verbose: bool = True):
    """Context manager to print start and end execution."""
    if verbose:
        print(f"\n{level}Begin calculating {argument}")
        yield
        print(f"{level}End calculating {argument}\n")
    else:
        yield
