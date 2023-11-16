import random
from collections import defaultdict
from typing import Callable, Generator, Iterable, Mapping, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")


# based on https://stackoverflow.com/a/480227/245362
def dedupe_stable(items: list[T]) -> list[T]:
    seen: set[T] = set()
    seen_add = seen.add
    return [item for item in items if not (item in seen or seen_add(item))]


def shallow_flatten(items: Iterable[Iterable[T]]) -> list[T]:
    return [item for sublist in items for item in sublist]


def batchify(
    data: Sequence[T], batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T], None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for i in tqdm(
        range(0, len(data), batch_size),
        total=(len(data) // batch_size + (len(data) % batch_size != 0)),
        disable=not show_progress,
    ):
        yield data[i : i + batch_size]


def tuplify(item: T | tuple[T, ...]) -> tuple[T, ...]:
    return item if isinstance(item, tuple) else (item,)


def stable_shuffle(items: list[T], seed: int | float | str = 42) -> list[T]:
    """
    Shuffle a list in a stable way
    """
    generator = random.Random(seed)
    # copy items to avoid modifying original
    results = [*items]
    generator.shuffle(results)
    return results


def stable_sample(items: list[T], k: int, seed: int | float | str = 42) -> list[T]:
    """
    Sample from a list in a stable way
    """
    generator = random.Random(seed)
    return generator.sample(items, k)


def find_all_substring_indices(
    string: str, substring: str, start: int = 0, end: int | None = None
) -> list[int]:
    """
    Find all indices of a substring in a string
    """
    indices = []
    while True:
        index = string.find(substring, start, end)
        if index == -1:
            break
        indices.append(index)
        start = index + len(substring)
    return indices


def sample_or_all(items: list[T], k: int, seed: int | float | str = 42) -> list[T]:
    """
    same as random.sample, but if k >= len(items), return items unmodified
    """
    generator = random.Random(seed)
    if k >= len(items):
        return items
    return generator.sample(items, k=k)


def mean(items: Sequence[float]) -> float:
    """
    Compute the mean of a list of numbers
    """
    return sum(items) / len(items)


def mean_values(items: Sequence[Mapping[T, float]]) -> dict[T, float]:
    """
    Compute the mean of a list of dicts of numbers
    """
    return {
        key: mean([item[key] for item in items if key in item])
        for key in set(key for item in items for key in item.keys())
    }


def group_items(items: Iterable[T], group_fn: Callable[[T], str]) -> dict[str, list[T]]:
    """
    Group items by the result of a function
    """
    grouped_items: dict[str, list[T]] = defaultdict(list)
    for item in items:
        group = group_fn(item)
        grouped_items[group].append(item)
    return grouped_items
