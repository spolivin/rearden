import functools
import time


def exec_timer(func):
    """Decorator for computing execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Time elasped:{execution_time: .1f} s")
        return result

    return wrapper
