import numpy as np

from time import time as t

from typing import List, Callable


def change_dtype_if_required(variable: np.ndarray,
                             dtype: type
                             ) -> np.ndarray:
    """Change the dtype of an array if necessary."""
    if variable.dtype != dtype:
        variable = variable.astype(dtype, copy=False)
    return variable


def change_several_dtype_if_required(list_variable: List,
                                     dtypes: List
                                     ) -> List:
    """Changes dtypes of several arrays if necessary."""
    result = []
    for variable, dtype in zip(list_variable, dtypes):
        if isinstance(variable, (list, int, float)):
            variable = np.array(variable)
        result.append(change_dtype_if_required(variable, dtype))
    return result


def print_func_executed_decorator(argument: str,
                                  level_begin: str = "",
                                  level_end: str = "",
                                  end: str = "",
                                  verbose: bool = True) -> Callable:
    """Decorator to print start and end of execution of a function."""
    def decorator(function):
        def wrapper(*args, **kwargs):
            if verbose:
                print(f"\n{level_begin}Begin {argument}")
            result = function(*args, **kwargs)
            if verbose:
                print(f"{level_end}End {argument}{end}\n")
            return result
        return wrapper
    return decorator


def timer_decorator(argument: str,
                    unit: str = 'minute',
                    level: str = "__",
                    verbose: bool = True,
                    ) -> Callable:
    """Decorator to print time of execution of a function."""
    def decorator(function):
        def wrapper(*args, **kwargs):
            if verbose:
                t0 = t()
            result = function(*args, **kwargs)
            if verbose:
                t1 = t()
                if unit == "hour":
                    time_execution = np.round((t1 - t0) / (3600), 2)
                elif unit == "minute":
                    time_execution = np.round((t1-t0) / 60, 2)
                elif unit == "second":
                    time_execution = np.round((t1 - t0), 2)
                print(f"{level}Time to calculate {argument}: {time_execution} {unit}s")
            return result
        return wrapper
    return decorator


def change_dtype_if_required_decorator(dtype: type) -> Callable:
    """Decorator to change dtype of the output of a function if necessary."""
    def decorator(function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)

            if isinstance(result, list):
                result = np.array(result)

            if result.dtype != dtype:
                result = result.astype(dtype, copy=False)

            return result
        return wrapper
    return decorator
