from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import torch
from tokenizers import Tokenizer
from torch import nn

from linear_relational.Concept import Concept
from linear_relational.lib.extract_token_activations import extract_token_activations
from linear_relational.lib.layer_matching import (
    LayerMatcher,
    collect_matching_layers,
    get_layer_name,
    guess_hidden_layer_matcher,
)
from linear_relational.lib.token_utils import (
    ensure_tokenizer_has_pad_token,
    find_final_word_token_index,
)
from linear_relational.lib.torch_utils import get_device
from linear_relational.lib.util import batchify

QuerySubject = Union[str, int, Callable[[str, list[int]], int]]


@dataclass
class ConceptMatchQuery:
    text: str
    subject: QuerySubject


@dataclass
class ConceptMatchResult:
    concept: str
    score: float


@dataclass
class QueryResult:
    concept_results: dict[str, ConceptMatchResult]

    @property
    def best_match(self) -> ConceptMatchResult:
        return max(self.concept_results.values(), key=lambda x: x.score)


class ConceptMatcher:
    concepts: list[Concept]
    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    layer_name_to_num: dict[str, int]

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        concepts: list[Concept],
        layer_matcher: Optional[LayerMatcher] = None,
    ) -> None:
        self.concepts = concepts
        self.model = model
        self.tokenizer = tokenizer
        self.layer_matcher = layer_matcher or guess_hidden_layer_matcher(model)
        ensure_tokenizer_has_pad_token(tokenizer)
        num_layers = len(collect_matching_layers(self.model, self.layer_matcher))
        self.layer_name_to_num = {}
        for layer_num in range(num_layers):
            self.layer_name_to_num[
                get_layer_name(model, self.layer_matcher, layer_num)
            ] = layer_num

    def query(self, query: str, subject: QuerySubject) -> QueryResult:
        return self.query_bulk([ConceptMatchQuery(query, subject)])[0]

    def query_bulk(
        self,
        queries: Sequence[ConceptMatchQuery],
        batch_size: int = 4,
        verbose: bool = False,
    ) -> list[QueryResult]:
        results: list[QueryResult] = []
        for batch in batchify(queries, batch_size, show_progress=verbose):
            results.extend(self._query_batch(batch))
        return results

    def _query_batch(self, queries: Sequence[ConceptMatchQuery]) -> list[QueryResult]:
        subj_tokens = [self._find_subject_token(query) for query in queries]
        with torch.no_grad():
            batch_subj_token_activations = extract_token_activations(
                self.model,
                self.tokenizer,
                layers=self.layer_name_to_num.keys(),
                texts=[q.text for q in queries],
                token_indices=subj_tokens,
                device=get_device(self.model),
                # batching is handled already, so no need to batch here too
                batch_size=len(queries),
                show_progress=False,
            )

        results: list[QueryResult] = []
        for raw_subj_token_activations in batch_subj_token_activations:
            concept_results: dict[str, ConceptMatchResult] = {}
            # need to replace the layer name with the layer number
            subj_token_activations = {
                self.layer_name_to_num[layer_name]: layer_activations[0]
                for layer_name, layer_activations in raw_subj_token_activations.items()
            }
            for concept in self.concepts:
                concept_results[concept.name] = _apply_concept_to_activations(
                    concept, subj_token_activations
                )
            results.append(QueryResult(concept_results))
        return results

    def _find_subject_token(self, query: ConceptMatchQuery) -> int:
        text = query.text
        subject = query.subject
        if isinstance(subject, int):
            return subject
        if isinstance(subject, str):
            return find_final_word_token_index(self.tokenizer, text, subject)
        if callable(subject):
            return subject(text, self.tokenizer.encode(text))
        raise ValueError(f"Unknown subject type: {type(subject)}")


@torch.no_grad()
def _apply_concept_to_activations(
    concept: Concept, activations: dict[int, torch.Tensor]
) -> ConceptMatchResult:
    score = concept.forward(activations[concept.layer]).item()
    return ConceptMatchResult(
        concept=concept.name,
        score=score,
    )
