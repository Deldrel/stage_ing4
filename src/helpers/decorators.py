from functools import wraps
from time import perf_counter


def timer(func: callable) -> callable:
    """
    Decorator to measure the execution time of a function.
    :param func: function to measure the execution time of
    :return: wrapper function that measures the execution time of the function
    """

    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"Execution time of {func.__name__}: {end - start:.6f} seconds")
        return result

    return wrapper


def try_except(func: callable) -> callable:
    """
    Decorator to catch exceptions in a function.
    :param func: function to catch exceptions in
    :return: wrapper function that catches exceptions in the function
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An exception occurred in {func.__name__}: {e}")

    return wrapper


def preserve_custom_attribute(attr_name: str):
    """
    Decorator to preserve a custom attribute of an object.
    :param attr_name: name of the custom attribute to preserve
    :return: decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(obj, *args, **kwargs):
            custom_attr = getattr(obj, attr_name, None)
            result_obj = func(obj, *args, **kwargs)
            if custom_attr is not None:
                setattr(result_obj, attr_name, custom_attr)
            return result_obj

        return wrapper

    return decorator
