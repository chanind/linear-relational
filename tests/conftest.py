import pytest
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, LlamaTokenizer

# loading in advance so it won't reload on every test
# just need to make sure not to edit these models in tests...
_model = GPT2LMHeadModel.from_pretrained("gpt2")
_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
_vicuna_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3", legacy=False)


@pytest.fixture
def model() -> GPT2LMHeadModel:
    return _model


@pytest.fixture
def tokenizer() -> GPT2TokenizerFast:
    return _tokenizer


@pytest.fixture
def vicuna_tokenizer() -> LlamaTokenizer:
    return _vicuna_tokenizer
