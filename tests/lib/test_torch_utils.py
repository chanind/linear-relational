from transformers import GPT2LMHeadModel

from linear_relational.lib.layer_matching import fix_neg_layer_num
from linear_relational.lib.torch_utils import guess_model_name


def test_guess_model_name(model: GPT2LMHeadModel) -> None:
    assert guess_model_name(model) == "gpt2"


def test_fix_neg_layer_num(model: GPT2LMHeadModel) -> None:
    assert fix_neg_layer_num(model, "transformer.h.{num}", -1) == 11
    assert fix_neg_layer_num(model, "transformer.h.{num}", -3) == 9
    assert fix_neg_layer_num(model, "transformer.h.{num}", 11) == 11
    assert fix_neg_layer_num(model, "transformer.h.{num}", 3) == 3
