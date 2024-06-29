from collections import OrderedDict

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from linear_relational.Concept import Concept
from linear_relational.ConceptMatcher import ConceptMatcher, ConceptMatchQuery
from linear_relational.lib.extract_token_activations import TokenLayerActivationsList
from tests.helpers import normalize


def test_ConceptMatcher_query(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    concept = Concept(
        object="test_object",
        relation="test_relation",
        layer=10,
        vector=torch.rand(768),
    )
    conceptifier = ConceptMatcher(
        model, tokenizer, [concept], layer_matcher="transformer.h.{num}"
    )
    result = conceptifier.query("This is a test", "test")
    assert len(result.concept_results) == 1
    assert concept.name in result.concept_results
    assert result.best_match.concept == concept.name


def test_ConceptMatcher_query_bulk(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    concept = Concept(
        object="test_object",
        relation="test_relation",
        layer=10,
        vector=torch.rand(768),
    )
    conceptifier = ConceptMatcher(
        model, tokenizer, [concept], layer_matcher="transformer.h.{num}"
    )
    results = conceptifier.query_bulk(
        [
            ConceptMatchQuery("This is a test", "test"),
            ConceptMatchQuery("This is another test", "test"),
        ]
    )
    assert len(results) == 2
    for result in results:
        assert concept.name in result.concept_results


def test_ConceptMatcher_query_bulk_with_map_activations_fn(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    concept_vec = normalize(torch.randn(768))
    concept1 = Concept(
        object="test_object1",
        relation="test_relation",
        layer=10,
        vector=concept_vec,
    )
    concept2 = Concept(
        object="test_object2",
        relation="test_relation",
        layer=10,
        vector=-1 * concept_vec,
    )

    def map_activations(
        token_layer_acts: TokenLayerActivationsList,
    ) -> TokenLayerActivationsList:
        mapped_token_layer_acts: TokenLayerActivationsList = []
        for token_layer_act in token_layer_acts:
            mapped_token_layer_act = OrderedDict()
            for layer, acts in token_layer_act.items():
                mapped_token_layer_act[layer] = [concept_vec for act in acts]
            mapped_token_layer_acts.append(mapped_token_layer_act)
        return mapped_token_layer_acts

    conceptifier = ConceptMatcher(
        model,
        tokenizer,
        [concept1, concept2],
        layer_matcher="transformer.h.{num}",
        map_activations_fn=map_activations,
    )
    results = conceptifier.query_bulk(
        [
            ConceptMatchQuery("This is a test", "test"),
            ConceptMatchQuery("This is another test", "test"),
        ]
    )
    assert len(results) == 2
    for result in results:
        assert result.concept_results[
            "test_relation: test_object1"
        ].score == pytest.approx(1.0)
        assert result.concept_results[
            "test_relation: test_object2"
        ].score == pytest.approx(-1.0)
