from collections.abc import Callable


def is_identical_function(callable_obj: Callable, target_function: Callable) -> bool:
    """Check if a callable object is the same function as a target function."""
    if not (callable(callable_obj) and callable(target_function)):
        return False

    # Unwrap bound callables (e.g. partials) to get the underlying function
    if hasattr(callable_obj, "__func__"):
        callable_obj = callable_obj.__func__
    if hasattr(target_function, "__func__"):
        target_function = target_function.__func__

    # Unwrap decorated functions to get the underlying function
    if hasattr(callable_obj, "__wrapped__"):
        callable_obj = callable_obj.__wrapped__
    if hasattr(target_function, "__wrapped__"):
        target_function = target_function.__wrapped__

    if hasattr(callable_obj, "__code__") and hasattr(target_function, "__code__"):
        return callable_obj.__code__ is target_function.__code__
    return False
