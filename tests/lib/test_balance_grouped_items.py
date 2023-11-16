from collections import defaultdict
from dataclasses import dataclass

from linear_relational.lib.balance_grouped_items import balance_grouped_items


@dataclass
class FakeItem:
    subject: str


items_by_group = {
    "London": [
        FakeItem(subject="London"),
        FakeItem(subject="London"),
        FakeItem(subject="London"),
        FakeItem(subject="London"),
    ],
    "Paris": [
        FakeItem(subject="Paris"),
        FakeItem(subject="Paris"),
        FakeItem(subject="Paris"),
        FakeItem(subject="Paris"),
    ],
    "Berlin": [
        FakeItem(subject="Berlin"),
        FakeItem(subject="Berlin"),
    ],
}


def test_balance_grouped_items_includes_all_items_with_no_limits_specified() -> None:
    items = balance_grouped_items(items_by_group)
    assert len(items) == 10
    counts_by_subj: dict[str, int] = defaultdict(int)
    for item in items:
        counts_by_subj[item.subject] += 1
    assert counts_by_subj == {
        "London": 4,
        "Paris": 4,
        "Berlin": 2,
    }


def test_balance_grouped_items_balances_all_items_until_the_max_is_hit() -> None:
    items = balance_grouped_items(items_by_group, max_total=6)
    assert len(items) == 6
    counts_by_subj: dict[str, int] = defaultdict(int)
    for item in items:
        counts_by_subj[item.subject] += 1
    assert counts_by_subj == {
        "London": 2,
        "Paris": 2,
        "Berlin": 2,
    }


def test_balance_grouped_items_balances_limits_the_amount_per_group() -> None:
    items = balance_grouped_items(items_by_group, max_per_group=3)
    assert len(items) == 8
    counts_by_subj: dict[str, int] = defaultdict(int)
    for item in items:
        counts_by_subj[item.subject] += 1
    assert counts_by_subj == {
        "London": 3,
        "Paris": 3,
        "Berlin": 2,
    }


def test_balance_grouped_items_with_total_and_per_group_limit() -> None:
    items = balance_grouped_items(items_by_group, max_per_group=3, max_total=100)
    assert len(items) == 8
    counts_by_subj: dict[str, int] = defaultdict(int)
    for item in items:
        counts_by_subj[item.subject] += 1
    assert counts_by_subj == {
        "London": 3,
        "Paris": 3,
        "Berlin": 2,
    }
