from collections import defaultdict
from time import time
from typing import Callable, Literal, Optional

import torch
from tokenizers import Tokenizer
from torch import nn

from linear_relational.Concept import Concept
from linear_relational.lib.balance_grouped_items import balance_grouped_items
from linear_relational.lib.extract_token_activations import extract_token_activations
from linear_relational.lib.layer_matching import (
    LayerMatcher,
    get_layer_name,
    guess_hidden_layer_matcher,
)
from linear_relational.lib.logger import log_or_print, logger
from linear_relational.lib.token_utils import PromptAnswerData, find_prompt_answer_data
from linear_relational.lib.torch_utils import get_device
from linear_relational.lib.util import group_items
from linear_relational.Lre import InvertedLre, Lre
from linear_relational.Prompt import Prompt
from linear_relational.PromptValidator import PromptValidator
from linear_relational.training.train_lre import ObjectAggregation, train_lre

VectorAggregation = Literal["pre_mean", "post_mean"]


class Trainer:
    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    prompt_validator: PromptValidator

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: Optional[LayerMatcher] = None,
        prompt_validator: Optional[PromptValidator] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_matcher = layer_matcher or guess_hidden_layer_matcher(model)
        self.prompt_validator = prompt_validator or PromptValidator(model, tokenizer)

    def train_lre(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        prompts: list[Prompt],
        object_aggregation: ObjectAggregation = "mean",
        validate_prompts: bool = True,
        validate_prompts_batch_size: int = 4,
        move_to_cpu: bool = False,
        verbose: bool = True,
    ) -> Lre:
        processed_prompts = self._process_relation_prompts(
            relation=relation,
            prompts=prompts,
            validate_prompts=validate_prompts,
            validate_prompts_batch_size=validate_prompts_batch_size,
            verbose=verbose,
        )
        return train_lre(
            model=self.model,
            tokenizer=self.tokenizer,
            layer_matcher=self.layer_matcher,
            relation=relation,
            subject_layer=subject_layer,
            object_layer=object_layer,
            prompts=processed_prompts,
            object_aggregation=object_aggregation,
            move_to_cpu=move_to_cpu,
        )

    def train_relation_concepts(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        prompts: list[Prompt],
        max_lre_training_samples: int | None = 20,
        object_aggregation: ObjectAggregation = "mean",
        vector_aggregation: VectorAggregation = "post_mean",
        inv_lre_rank: int = 200,
        validate_prompts_batch_size: int = 4,
        validate_prompts: bool = True,
        verbose: bool = True,
        name_concept_fn: Optional[Callable[[str, str], str]] = None,
        seed: int | str | float = 42,
    ) -> list[Concept]:
        processed_prompts = self._process_relation_prompts(
            relation=relation,
            prompts=prompts,
            validate_prompts=validate_prompts,
            validate_prompts_batch_size=validate_prompts_batch_size,
            verbose=verbose,
        )
        prompts_by_object = group_items(processed_prompts, lambda p: p.object_name)
        if len(prompts_by_object) == 1:
            logger.warning(
                f"Only one valid object found for {relation}. Results may be poor."
            )
        lre_train_prompts = balance_grouped_items(
            items_by_group=prompts_by_object,
            max_total=max_lre_training_samples,
            seed=seed,
        )
        inv_lre = self.train_lre(
            relation=relation,
            subject_layer=subject_layer,
            object_layer=object_layer,
            prompts=lre_train_prompts,
            object_aggregation=object_aggregation,
            validate_prompts=False,  # we already validated the prompts above
            validate_prompts_batch_size=validate_prompts_batch_size,
            verbose=verbose,
        ).invert(inv_lre_rank)

        return self.train_relation_concepts_from_inv_lre(
            inv_lre=inv_lre,
            prompts=processed_prompts,
            vector_aggregation=vector_aggregation,
            validate_prompts_batch_size=validate_prompts_batch_size,
            validate_prompts=False,  # we already validated the prompts above
            name_concept_fn=name_concept_fn,
            verbose=verbose,
        )

    def train_relation_concepts_from_inv_lre(
        self,
        inv_lre: InvertedLre,
        prompts: list[Prompt],
        vector_aggregation: VectorAggregation = "post_mean",
        validate_prompts_batch_size: int = 4,
        extract_objects_batch_size: int = 4,
        validate_prompts: bool = True,
        name_concept_fn: Optional[Callable[[str, str], str]] = None,
        verbose: bool = True,
    ) -> list[Concept]:
        relation = inv_lre.relation
        processed_prompts = self._process_relation_prompts(
            relation=relation,
            prompts=prompts,
            validate_prompts=validate_prompts,
            validate_prompts_batch_size=validate_prompts_batch_size,
            verbose=verbose,
        )
        start_time = time()
        object_activations = self._extract_target_object_activations_for_inv_lre(
            prompts=processed_prompts,
            batch_size=extract_objects_batch_size,
            object_aggregation=inv_lre.object_aggregation,
            object_layer=inv_lre.object_layer,
            show_progress=verbose,
            move_to_cpu=True,
        )
        logger.info(
            f"Extracted {len(object_activations)} object activations in {time() - start_time:.2f}s"
        )
        concepts: list[Concept] = []

        with torch.no_grad():
            for (
                object_name,
                activations,
            ) in object_activations.items():
                name = None
                if name_concept_fn is not None:
                    name = name_concept_fn(relation, object_name)
                concept = self._build_concept(
                    relation_name=relation,
                    layer=inv_lre.subject_layer,
                    inv_lre=inv_lre,
                    object_name=object_name,
                    activations=activations,
                    vector_aggregation=vector_aggregation,
                    name=name,
                )
                concepts.append(concept)
        return concepts

    def _process_relation_prompts(
        self,
        relation: str,
        prompts: list[Prompt],
        validate_prompts: bool,
        validate_prompts_batch_size: int,
        verbose: bool,
    ) -> list[Prompt]:
        valid_prompts = prompts
        if validate_prompts:
            log_or_print(f"validating {len(prompts)} prompts", verbose=verbose)
            valid_prompts = self.prompt_validator.filter_prompts(
                prompts, validate_prompts_batch_size, verbose
            )
        if len(valid_prompts) == 0:
            raise ValueError(f"No valid prompts found for {relation}.")
        return valid_prompts

    def _build_concept(
        self,
        layer: int,
        relation_name: str,
        object_name: str,
        activations: list[torch.Tensor],
        inv_lre: InvertedLre,
        vector_aggregation: VectorAggregation,
        name: str | None,
    ) -> Concept:
        device = inv_lre.bias.device
        dtype = inv_lre.bias.dtype
        if vector_aggregation == "pre_mean":
            acts = [torch.stack(activations).to(device=device, dtype=dtype).mean(dim=0)]
        elif vector_aggregation == "post_mean":
            acts = [act.to(device=device, dtype=dtype) for act in activations]
        else:
            raise ValueError(f"Unknown vector aggregation method {vector_aggregation}")
        vecs = [
            inv_lre.calculate_subject_activation(act, normalize=False) for act in acts
        ]
        vec = torch.stack(vecs).mean(dim=0)
        vec = vec / vec.norm()
        return Concept(
            name=name,
            object=object_name,
            relation=relation_name,
            layer=layer,
            vector=vec.detach().clone().cpu(),
        )

    @torch.no_grad()
    def _extract_target_object_activations_for_inv_lre(
        self,
        object_layer: int,
        object_aggregation: Literal["mean", "first_token"],
        prompts: list[Prompt],
        batch_size: int,
        show_progress: bool = False,
        move_to_cpu: bool = True,
    ) -> dict[str, list[torch.Tensor]]:
        activations_by_object: dict[str, list[torch.Tensor]] = defaultdict(list)
        prompt_answer_data: list[PromptAnswerData] = []
        for prompt in prompts:
            prompt_answer_data.append(
                find_prompt_answer_data(self.tokenizer, prompt.text, prompt.answer)
            )

        layer_name = get_layer_name(self.model, self.layer_matcher, object_layer)
        raw_activations = extract_token_activations(
            self.model,
            self.tokenizer,
            layers=[layer_name],
            texts=[prompt_answer.full_prompt for prompt_answer in prompt_answer_data],
            token_indices=[
                prompt_answer.output_answer_token_indices
                for prompt_answer in prompt_answer_data
            ],
            device=get_device(self.model),
            batch_size=batch_size,
            show_progress=show_progress,
            move_results_to_cpu=move_to_cpu,
        )
        for prompt, raw_activation in zip(prompts, raw_activations):
            if object_aggregation == "mean":
                activation = torch.stack(raw_activation[layer_name]).mean(dim=0)
            elif object_aggregation == "first_token":
                activation = raw_activation[layer_name][0]
            else:
                raise ValueError(
                    f"Unknown inv_lre.object_aggregation: {object_aggregation}"
                )
            activations_by_object[prompt.object_name].append(activation)
        return activations_by_object
