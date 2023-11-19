from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from linear_relational.Prompt import Prompt
from linear_relational.PromptValidator import PromptValidator, cache_key


def test_cache_key() -> None:
    assert cache_key("blah this is a prompt", "answer") == "474096ceddcbb81"


def test_filter_prompts(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> None:
    prompts = [
        Prompt(
            text="Rome is located in the country of",
            answer="Italy",
            subject="Rome",
        ),
        Prompt(
            text="Beijing is located in the country of",
            answer="China",
            subject="Beijing",
        ),
        Prompt(
            text="Fakeplace is located in the country of",
            answer="Nowhereland",
            subject="Fakeplace",
        ),
    ]
    validator = PromptValidator(model, tokenizer)
    filtered_prompts = validator.filter_prompts(prompts)
    print(filtered_prompts)
    assert len(filtered_prompts) == 2
    assert filtered_prompts[0].subject == "Rome"
    assert filtered_prompts[1].subject == "Beijing"
