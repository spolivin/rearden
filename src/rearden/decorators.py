"""Tools for decorating functions."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import functools
import time
from typing import Any, Callable

import numpy as np
from sklearn.exceptions import NotFittedError


class DataSplitterDecorators:
    """Class for decorators for DataSplitter class."""

    @staticmethod
    def check_proportions(attr: str) -> Callable:
        """Checks the correctness of set shares in DataSplitter object."""

        def inner(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(self, *args: Any, **kwargs: Any) -> Callable:
                # Getting the required attribute value
                set_shares = getattr(self, attr)
                # Computing the sum of set shares
                set_shares_sum = np.sum(set_shares)
                set_shares_sum_rnd = np.round(set_shares_sum, 2)
                # Verifying the consistency of shares
                if set_shares_sum_rnd != 1.0:
                    # Raising error in case sum of shares is higher than one
                    if set_shares_sum_rnd > 1.0:
                        runtime_err_msg = (
                            "Incorrect proportions specified. "
                            f"Sum of shares in '{attr}' is higher than one."
                        )
                    # Raising error in case sum of shares is less than one
                    else:
                        runtime_err_msg = (
                            "Incorrect proportions specified. "
                            f"Sum of shares in '{attr}' is less than one."
                        )
                    raise RuntimeError(runtime_err_msg)
                # Case when sum of shares is one due to negative numbers passed
                lst = [set_share for set_share in set_shares if set_share <= 0]
                if lst != []:
                    runtime_err_msg = f"'{attr}' takes only positive values."
                    raise RuntimeError(runtime_err_msg)

                return func(self, *args, **kwargs)

            return wrapper

        return inner

    @staticmethod
    def check_dimensions(attr: str) -> Callable:
        """Checks the correctness of set shares number in DataSplitter object."""

        def inner(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(self, *args: Any, **kwargs: Any) -> Callable:
                # Getting the value of the attribute
                set_shares = getattr(self, attr)
                # Verifying the number of shares passed
                try:
                    set_shares_num = len(set_shares)
                except TypeError:
                    set_shares_num = 1
                # Raising error is the number of shares is not 2 or 3
                if set_shares_num not in (2, 3):
                    runtime_err_msg = (
                        f"'{attr}' argument cannot have a length of {set_shares_num}. "
                        "Acceptable length is 2 or 3."
                    )
                    raise RuntimeError(runtime_err_msg)
                return func(self, *args, **kwargs)

            return wrapper

        return inner

    @staticmethod
    def check_split(attr: str, func_hint: str) -> Callable:
        """Checks if the data has been split."""

        def inner(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(self, *args: Any, **kwargs: Any) -> Callable:
                # Raising error is objects does not have `attr` attribute (data unsplit)
                if not hasattr(self, attr):
                    attr_err_msg = (
                        "Data has not been split. "
                        f"Call '{func_hint}' before accessing this attribute."
                    )
                    raise AttributeError(attr_err_msg)
                return func(self, *args, **kwargs)

            return wrapper

        return inner


def exec_timer(func: Callable) -> Callable:
    """Decorator for computing execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Callable:
        # Start of function execution time measurement
        start_time = time.time()
        # Running the decorated function
        result = func(*args, **kwargs)
        # Computing function execution time
        end_time = time.time()
        # Displaying the elapsed time
        execution_time = end_time - start_time
        print(f"Time elapsed:{execution_time: .1f} s")

        return result

    return wrapper


def check_est_fit_by_attr(attr: str, func_hint: str) -> Callable:
    """Checks if the estimator has been fit using an attribute."""

    def inner(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Callable:
            # Checking if an instance has the passed attribute
            if not hasattr(self, attr):
                not_fitted_err_msg = (
                    f"The {type(self).__name__} instance has not been fitted yet. "
                    f"Call '{func_hint}' before using this method."
                )
                raise NotFittedError(not_fitted_err_msg)
            # Running the function if exception has not been raised
            return func(self, *args, **kwargs)

        return wrapper

    return inner
