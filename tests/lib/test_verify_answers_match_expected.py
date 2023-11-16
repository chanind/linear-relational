from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from linear_relational.lib.verify_answers_match_expected import (
    verify_answers_match_expected,
)


def test_verify_answers_match_expected(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    results = verify_answers_match_expected(
        model,
        tokenizer,
        [
            "Bill Gates is the CEO of",
            "Tokyo is located in the country of",
            "Steve Jobs is the CEO of",
        ],
        [(" Microsoft", " Microsoft Corporation"), (" Japan",), (" Orange",)],
    )

    assert [res.answer_matches_expected for res in results] == [True, True, False]
    assert [res.prompt for res in results] == [
        "Bill Gates is the CEO of",
        "Tokyo is located in the country of",
        "Steve Jobs is the CEO of",
    ]
    assert results[0].potential_answers == {" Microsoft", " Microsoft,"}
    assert results[1].potential_answers == {" Japan", " Japan,"}
    assert results[2].potential_answers == {" Apple", " Apple."}


def test_verify_answers_match_expected_handles_space_normalization(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    results = verify_answers_match_expected(
        model,
        tokenizer,
        [
            "Bill Gates is the CEO of",
            "Tokyo is located in the country of",
            "Steve Jobs is the CEO of",
        ],
        [("Microsoft", "Microsoft Corporation"), ("Japan",), ("Orange",)],
    )
    assert [res.answer_matches_expected for res in results] == [True, True, False]


def test_verify_answers_match_allows_passing_single_strings(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    results = verify_answers_match_expected(
        model,
        tokenizer,
        [
            "Bill Gates is the CEO of",
            "Tokyo is located in the country of",
            "Steve Jobs is the CEO of",
        ],
        [
            ("Microsoft", "Microsoft Corporation"),
            "Japan",
            "Orange",
        ],
    )
    assert [res.answer_matches_expected for res in results] == [True, True, False]
