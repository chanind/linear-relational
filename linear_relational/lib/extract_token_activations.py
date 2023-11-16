from collections import OrderedDict
from typing import Iterable, Sequence

import torch
from tokenizers import Tokenizer
from torch import nn

from .constants import DEFAULT_DEVICE
from .token_utils import find_final_word_token_index, make_inputs
from .torch_utils import untuple_tensor
from .TraceLayerDict import TraceLayerDict
from .util import batchify, tuplify


def extract_token_activations(
    model: nn.Module,
    tokenizer: Tokenizer,
    layers: Iterable[str],
    texts: Sequence[str],
    token_indices: Sequence[tuple[int, ...] | int],
    device: torch.device = DEFAULT_DEVICE,
    move_results_to_cpu: bool = True,
    batch_size: int = 32,
    show_progress: bool = False,
) -> list[OrderedDict[str, list[torch.Tensor]]]:
    if len(texts) != len(token_indices):
        raise ValueError(
            f"Expected {len(texts)} texts to match {len(token_indices)} subject token indices"
        )
    results: list[OrderedDict[str, list[torch.Tensor]]] = []
    for batch in batchify(
        # need to turn the zip into a list or mypy complains
        list(zip(texts, token_indices)),
        batch_size=batch_size,
        show_progress=show_progress,
    ):
        batch_texts = [t for t, _ in batch]
        batch_subject_token_indices = [tuplify(indices) for _, indices in batch]
        batch_subj_token_activations = _extract_token_activations_batch(
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            texts=batch_texts,
            token_indices=batch_subject_token_indices,
            device=device,
            move_results_to_cpu=move_results_to_cpu,
        )
        results.extend(batch_subj_token_activations)
    return results


def _extract_token_activations_batch(
    model: nn.Module,
    tokenizer: Tokenizer,
    layers: Iterable[str],
    texts: list[str],
    token_indices: list[tuple[int, ...]],
    device: torch.device = DEFAULT_DEVICE,
    move_results_to_cpu: bool = True,
) -> list[OrderedDict[str, list[torch.Tensor]]]:
    if len(texts) != len(token_indices):
        raise ValueError(
            f"Expected {len(texts)} texts to match {len(token_indices)} subject token indices"
        )
    inputs = make_inputs(
        tokenizer=tokenizer,
        prompts=texts,
        device=device,
    )
    batch_token_activations: list[OrderedDict[str, list[torch.Tensor]]] = [
        OrderedDict() for _ in texts
    ]
    with TraceLayerDict(
        model,
        layers=layers,
        retain_output=True,
    ) as td:
        model(**inputs)
        for layer_name, layer_trace in td.items():
            assert layer_trace.output is not None
            raw_output = untuple_tensor(layer_trace.output).detach()
            for i, toks in enumerate(token_indices):
                activations = []
                for tok in toks:
                    activation = raw_output[i, tok].clone().detach().type(torch.float32)
                    if move_results_to_cpu:
                        activation = activation.cpu()
                    activations.append(activation)
                batch_token_activations[i][layer_name] = activations
    return batch_token_activations


def extract_final_token_activations(
    model: nn.Module,
    tokenizer: Tokenizer,
    texts: Sequence[str],
    layers: Iterable[str],
    device: torch.device = DEFAULT_DEVICE,
    move_results_to_cpu: bool = True,
    batch_size: int = 32,
    show_progress: bool = False,
) -> list[OrderedDict[str, torch.Tensor]]:
    raw_activations = extract_token_activations(
        model,
        tokenizer,
        layers=layers,
        texts=texts,
        token_indices=[
            find_final_word_token_index(tokenizer, text, text) for text in texts
        ],
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
        move_results_to_cpu=move_results_to_cpu,
    )
    return [_pick_first_activation(activations) for activations in raw_activations]


def _pick_first_activation(
    token_activations: OrderedDict[str, list[torch.Tensor]]
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (layer_name, layer_activations[0])
        for layer_name, layer_activations in token_activations.items()
    )
