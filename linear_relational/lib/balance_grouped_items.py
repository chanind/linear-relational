from __future__ import annotations

from collections import defaultdict
from typing import Optional, TypeVar

from linear_relational.lib.util import stable_shuffle

T = TypeVar("T")


def balance_grouped_items(
    items_by_group: dict[str, list[T]],
    max_per_group: Optional[int] = None,
    max_total: Optional[int] = None,
    seed: int | float | str = 42,
) -> list[T]:
    """
    Pick items in a round-robin fashion from each of the possible groups
    Tries to balance the amount of items that come from each group as much as possible
    `items_by_group` is a dict of group name to list of items
    """
    requests: list[T] = []
    concept_names = stable_shuffle(list(items_by_group.keys()), seed=seed)
    shuffled_reqs_by_concept = {
        concept: stable_shuffle(reqs, seed=f"{seed}{concept}")
        for concept, reqs in items_by_group.items()
    }
    prompts_per_concept: dict[str, int] = defaultdict(int)
    total_prompts = 0
    for reqs in items_by_group.values():
        num_reqs_from_concept = len(reqs)
        if max_per_group is not None and num_reqs_from_concept > max_per_group:
            num_reqs_from_concept = max_per_group
        total_prompts += num_reqs_from_concept
    if max_total is not None:
        total_prompts = min(total_prompts, max_total)

    concept_index = 0
    while len(requests) < total_prompts:
        concept_name = concept_names[concept_index]
        reqs = shuffled_reqs_by_concept[concept_name]
        concept_index = (concept_index + 1) % len(concept_names)
        if (
            max_per_group is not None
            and prompts_per_concept[concept_name] >= max_per_group
        ):
            continue
        if prompts_per_concept[concept_name] >= len(reqs):
            continue
        requests.append(reqs[prompts_per_concept[concept_name]])
        prompts_per_concept[concept_name] += 1
    return requests
