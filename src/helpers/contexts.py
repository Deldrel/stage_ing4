import contextlib
import random
from typing import Any, Generator, Optional


@contextlib.contextmanager
def rand_state(seed: Optional[int] = None) -> Generator:
    """
    Context manager to manipulate the state of the random number generator.
    The original state is restored when the context manager exits. (seed only works inside the context)
    :param seed: the seed to set the random number generator to
    :return: Generator
    """
    state: tuple[Any, ...] = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)
