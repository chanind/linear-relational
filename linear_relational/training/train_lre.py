from typing import Iterable, Literal

import torch
from tokenizers import Tokenizer
from torch import nn

from linear_relational.lib.layer_matching import (
    LayerMatcher,
    fix_neg_layer_num,
    get_layer_name,
)
from linear_relational.lib.token_utils import (
    find_final_word_token_index,
    find_prompt_answer_data,
)
from linear_relational.lib.torch_utils import get_device, untuple_tensor
from linear_relational.lib.TraceLayer import TraceLayer
from linear_relational.lib.TraceLayerDict import TraceLayerDict
from linear_relational.Lre import Lre
from linear_relational.Prompt import Prompt

ObjectAggregation = Literal["mean", "first_token"]


def train_lre(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    relation: str,
    subject_layer: int,
    object_layer: int,
    prompts: list[Prompt],
    object_aggregation: ObjectAggregation = "mean",
    move_to_cpu: bool = False,
) -> Lre:
    weights = []
    biases = []
    full_prompts = []
    object_layer = fix_neg_layer_num(model, layer_matcher, object_layer)
    for prompt in prompts:
        prompt_answer_data = find_prompt_answer_data(
            tokenizer, prompt.text, prompt.answer
        )
        full_prompts.append(prompt_answer_data.full_prompt)
        subject_index = find_final_word_token_index(
            tokenizer, prompt.text, prompt.subject
        )
        weight, bias = order_1_approx(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_answer_data.full_prompt,
            subject_layer_name=get_layer_name(model, layer_matcher, subject_layer),
            object_layer_name=get_layer_name(model, layer_matcher, object_layer),
            subject_index=subject_index,
            object_pred_indices=prompt_answer_data.output_answer_token_indices,
            object_aggregation=object_aggregation,
        )
        weights.append(weight)
        biases.append(bias)
    weight = torch.stack(weights).mean(dim=0).detach()
    bias = torch.stack(biases).mean(dim=0).squeeze(0).detach()
    if move_to_cpu:
        weight = weight.cpu()
        bias = bias.cpu()

    return Lre(
        relation=relation,
        subject_layer=subject_layer,
        object_layer=object_layer,
        weight=weight,
        bias=bias,
        object_aggregation=object_aggregation,
        metadata={"num_prompts": len(prompts)},
    )


# Heavily based on https://github.com/evandez/relations/blob/main/src/functional.py


@torch.no_grad()
@torch.inference_mode(mode=False)
def order_1_approx(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt_text: str,
    subject_layer_name: str,
    object_layer_name: str,
    subject_index: int,
    object_pred_indices: Iterable[int],
    object_aggregation: ObjectAggregation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a first-order approximation of the LM between `subject` and `object`.

    Very simply, this computes the Jacobian of object with respect to subject, as well as
    object - (J * subject) to approximate the bias.

    Returns the weight and bias

    This is an adapted version of order_1_approx from https://github.com/evandez/relations/blob/main/src/functional.py
    """
    device = get_device(model)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Precompute everything up to the subject, if there is anything before it.
    past_key_values = None
    input_ids = inputs.input_ids
    _subject_index = subject_index
    _object_pred_indices = object_pred_indices
    if _subject_index > 0:
        outputs = model(input_ids=input_ids[:, :_subject_index], use_cache=True)
        past_key_values = outputs.past_key_values
        input_ids = input_ids[:, _subject_index:]
        _subject_index = 0
        _object_pred_indices = [i - subject_index for i in object_pred_indices]
    use_cache = past_key_values is not None

    # Precompute initial h and z.
    with TraceLayerDict(
        model, layers=(subject_layer_name, object_layer_name), stop=True
    ) as ret:
        model(
            input_ids=input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
    subject_layer_output = ret[subject_layer_name].output
    assert subject_layer_output is not None  # keep mypy happy
    subject_activation = untuple_tensor(subject_layer_output)[0, _subject_index]
    object_activation = _extract_object_activation(
        ret[object_layer_name], _object_pred_indices, object_aggregation
    )

    # Now compute J and b.
    def compute_object_from_subject(subject_activation: torch.Tensor) -> torch.Tensor:
        def insert_h(
            output: tuple[torch.Tensor, ...], layer: str
        ) -> tuple[torch.Tensor, ...]:
            hs = untuple_tensor(output)
            if layer != subject_layer_name:
                return output
            hs[0, _subject_index] = subject_activation
            return output

        with TraceLayerDict(
            model,
            (subject_layer_name, object_layer_name),
            edit_output=insert_h,
            stop=True,
        ) as ret:
            model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return _extract_object_activation(
            ret[object_layer_name], _object_pred_indices, object_aggregation
        )

    assert subject_activation is not None
    weight = torch.autograd.functional.jacobian(
        compute_object_from_subject, subject_activation, vectorize=True
    )
    bias = object_activation[None] - subject_activation[None].mm(weight.t())

    # NB(evan): Something about the jacobian computation causes a lot of memory
    # fragmentation, or some kind of memory leak. This seems to help.
    torch.cuda.empty_cache()

    return weight.detach(), bias.detach()


def _extract_object_activation(
    object_layer_trace: TraceLayer,
    object_pred_indices: Iterable[int],
    object_aggregation: Literal["mean", "first_token"],
) -> torch.Tensor:
    object_layer_output = object_layer_trace.output
    assert object_layer_output is not None  # keep mypy happy
    object_activations = untuple_tensor(object_layer_output)[0, object_pred_indices]
    if object_aggregation == "mean":
        return object_activations.mean(dim=0)
    elif object_aggregation == "first_token":
        return object_activations[0]
