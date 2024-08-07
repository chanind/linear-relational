import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, PreTrainedModel

from linear_relational.lib.extract_token_activations import (
    extract_final_token_activations,
    extract_token_activations,
)
from linear_relational.lib.token_utils import find_token_range
from linear_relational.training.train_lre import train_lre
from tests.helpers import create_prompt


def test_train_lre(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> None:
    fsl_prefixes = "\n".join(
        [
            "Berlin is located in the country of Germany",
            "Toronto is located in the country of Canada",
            "Lagos is located in the country of Nigeria",
        ]
    )
    prompts = [
        create_prompt(
            text=f"{fsl_prefixes}\nTokyo is located in the country of",
            answer="Japan",
            subject="Tokyo",
        ),
        create_prompt(
            text=f"{fsl_prefixes}\nRome is located in the country of",
            answer="Italy",
            subject="Rome",
        ),
    ]
    lre = train_lre(
        model=model,
        tokenizer=tokenizer,
        layer_matcher="transformer.h.{num}",
        relation="city in country",
        subject_layer=5,
        object_layer=9,
        prompts=prompts,
        object_aggregation="mean",
    )
    assert lre.relation == "city in country"
    assert lre.subject_layer == 5
    assert lre.object_layer == 9
    assert lre.weight.shape == (768, 768)
    assert lre.bias.shape == (768,)


def test_train_lre_on_single_prompt_perfectly_replicates_object(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    fsl_prefixes = "\n".join(
        [
            "Berlin is located in the country of Germany",
            "Toronto is located in the country of Canada",
            "Lagos is located in the country of Nigeria",
        ]
    )
    prompt = create_prompt(
        text=f"{fsl_prefixes}\nTokyo is located in the country of",
        answer="Japan",
        subject="Tokyo",
    )
    prompts = [prompt]
    lre = train_lre(
        model=model,
        tokenizer=tokenizer,
        layer_matcher="transformer.h.{num}",
        relation="city in country",
        subject_layer=5,
        object_layer=9,
        prompts=prompts,
        object_aggregation="mean",
    )

    subj_index = (
        find_token_range(tokenizer, tokenizer.encode(prompt.text), prompt.subject)[-1]
        - 1
    )
    subj_act = extract_token_activations(
        model=model,
        tokenizer=tokenizer,
        texts=[prompt.text],
        layers=["transformer.h.5"],
        token_indices=[subj_index],
    )[0]["transformer.h.5"][0]
    obj_act = extract_final_token_activations(
        model=model,
        tokenizer=tokenizer,
        texts=[prompt.text],
        layers=["transformer.h.9"],
    )[0]["transformer.h.9"]
    assert torch.allclose(lre(subj_act), obj_act, atol=1e-4)


def test_train_lre_on_single_prompt_with_gemma2_perfectly_replicates_object(
    empty_gemma2_model: PreTrainedModel, tokenizer: GPT2TokenizerFast
) -> None:
    fsl_prefixes = "\n".join(
        [
            "Berlin is located in the country of Germany",
            "Toronto is located in the country of Canada",
            "Lagos is located in the country of Nigeria",
        ]
    )
    prompt = create_prompt(
        text=f"{fsl_prefixes}\nTokyo is located in the country of",
        answer="Japan",
        subject="Tokyo",
    )
    prompts = [prompt]
    lre = train_lre(
        model=empty_gemma2_model,
        tokenizer=tokenizer,
        layer_matcher="model.layers.{num}",
        relation="city in country",
        subject_layer=1,
        object_layer=2,
        prompts=prompts,
        object_aggregation="mean",
    )

    subj_index = (
        find_token_range(tokenizer, tokenizer.encode(prompt.text), prompt.subject)[-1]
        - 1
    )
    subj_act = extract_token_activations(
        model=empty_gemma2_model,
        tokenizer=tokenizer,
        texts=[prompt.text],
        layers=["model.layers.1"],
        token_indices=[subj_index],
    )[0]["model.layers.1"][0]
    obj_act = extract_final_token_activations(
        model=empty_gemma2_model,
        tokenizer=tokenizer,
        texts=[prompt.text],
        layers=["model.layers.2"],
    )[0]["model.layers.2"]
    assert torch.allclose(lre(subj_act), obj_act, atol=1e-4)


def test_train_lre_works_with_gemma2_and_float16(
    empty_gemma2_model: PreTrainedModel, tokenizer: GPT2TokenizerFast
) -> None:
    model = empty_gemma2_model.half()
    prompt = create_prompt(
        text="Tokyo is located in the country of",
        answer="Japan",
        subject="Tokyo",
    )
    lre = train_lre(
        model=model,
        tokenizer=tokenizer,
        layer_matcher="model.layers.{num}",
        relation="city in country",
        subject_layer=1,
        object_layer=2,
        prompts=[prompt],
    ).float()

    subj_index = (
        find_token_range(tokenizer, tokenizer.encode(prompt.text), prompt.subject)[-1]
        - 1
    )
    subj_act = extract_token_activations(
        model=model,
        tokenizer=tokenizer,
        texts=[prompt.text],
        layers=["model.layers.1"],
        token_indices=[subj_index],
    )[0]["model.layers.1"][0]
    obj_act = extract_final_token_activations(
        model=model,
        tokenizer=tokenizer,
        texts=[prompt.text],
        layers=["model.layers.2"],
    )[0]["model.layers.2"]
    assert torch.allclose(lre(subj_act), obj_act, atol=5e-4)
