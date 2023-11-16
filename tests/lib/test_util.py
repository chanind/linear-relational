from linear_relational.lib.util import (
    dedupe_stable,
    find_all_substring_indices,
    stable_shuffle,
)


def test_dedule_stable() -> None:
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert dedupe_stable(items) == items
    assert dedupe_stable(items + items) == items
    assert dedupe_stable(items + items + items) == items


def test_stable_shuffle() -> None:
    original = [1, 2, 3, 4, 5]
    shuffled1 = stable_shuffle(original, seed=42)
    shuffled2 = stable_shuffle(original, seed=42)
    shuffled3 = stable_shuffle(original, seed=123)
    assert original == [1, 2, 3, 4, 5]
    assert shuffled1 != original
    assert shuffled3 != original
    assert shuffled1 == shuffled2
    assert shuffled1 != shuffled3


def test_find_all_substring_indices() -> None:
    assert find_all_substring_indices("Hello, World!", "l") == [2, 3, 10]
    assert find_all_substring_indices("Hello, World!", "l", 3) == [3, 10]
    assert find_all_substring_indices("Hello, World!", "l", 3, 9) == [3]
    assert find_all_substring_indices("Hello, World!", "Hello") == [0]
