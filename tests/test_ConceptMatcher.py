import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from linear_relational.Concept import Concept
from linear_relational.ConceptMatcher import ConceptMatcher, ConceptMatchQuery


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
