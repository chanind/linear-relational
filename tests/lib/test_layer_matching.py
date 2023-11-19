from transformers import GPT2LMHeadModel

from linear_relational.lib.layer_matching import (
    _guess_hidden_layer_matcher_from_layers,
    guess_hidden_layer_matcher,
)


def test_guess_hidden_layer_matcher(model: GPT2LMHeadModel) -> None:
    assert guess_hidden_layer_matcher(model) == "transformer.h.{num}"


def test_guess_hidden_layer_matcher_from_layers() -> None:
    layers = [
        "x.e",
        "x.y.0",
        "x.y.0.attn",
        "x.y.1",
        "x.y.1.attn",
        "x.y.2",
        "x.y.2.attn",
        "x.lm_head",
    ]
    assert _guess_hidden_layer_matcher_from_layers(layers) == "x.y.{num}"


def test_guess_hidden_layer_matcher_from_layers_guess_llama_matcher() -> None:
    layers = [
        "",
        "model",
        "model.embed_tokens",
        "model.layers",
        "model.layers.0",
        "model.layers.0.self_attn",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.rotary_emb",
        "model.layers.0.mlp",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.act_fn",
        "model.layers.0.input_layernorm",
        "model.layers.0.post_attention_layernorm",
        "model.layers.1",
        "model.layers.1.self_attn",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.1.self_attn.rotary_emb",
        "model.layers.1.mlp",
        "model.layers.1.mlp.gate_proj",
        "model.layers.1.mlp.up_proj",
        "model.layers.1.mlp.down_proj",
        "model.layers.1.mlp.act_fn",
        "model.layers.1.input_layernorm",
        "model.layers.1.post_attention_layernorm",
        "model.layers.2",
        "model.layers.2.self_attn",
        "model.layers.2.self_attn.q_proj",
        "model.layers.2.self_attn.k_proj",
        "model.layers.2.self_attn.v_proj",
        "model.layers.2.self_attn.o_proj",
        "model.layers.2.self_attn.rotary_emb",
        "model.layers.2.mlp",
        "model.layers.2.mlp.gate_proj",
        "model.layers.2.mlp.up_proj",
        "model.layers.2.mlp.down_proj",
        "model.layers.2.mlp.act_fn",
        "model.layers.2.input_layernorm",
        "model.layers.2.post_attention_layernorm",
        "model.layers.3",
        "model.layers.3.self_attn",
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.self_attn.k_proj",
        "model.layers.3.self_attn.v_proj",
        "model.layers.3.self_attn.o_proj",
        "model.layers.3.self_attn.rotary_emb",
        "model.layers.3.mlp",
        "model.layers.3.mlp.gate_proj",
        "model.layers.3.mlp.up_proj",
        "model.layers.3.mlp.down_proj",
        "model.layers.3.mlp.act_fn",
        "model.layers.3.input_layernorm",
        "model.layers.3.post_attention_layernorm",
        "model.norm",
        "lm_head",
    ]
    assert _guess_hidden_layer_matcher_from_layers(layers) == "model.layers.{num}"
