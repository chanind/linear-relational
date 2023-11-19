import pytest
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from linear_relational.lib.util import stable_shuffle
from linear_relational.training.Trainer import Trainer
from tests.helpers import create_prompt


def fsl_prefix(
    target_sample: tuple[str, str],
    samples: list[tuple[str, str]],
    template: str,
    num_prefix_samples: int = 4,
) -> str:
    prefixes: list[str] = []
    samples_cpy = [*samples]
    while len(prefixes) < num_prefix_samples:
        prefix_sample = samples_cpy.pop()
        if prefix_sample != target_sample:
            prefixes.append(template.format(prefix_sample[0]) + " " + prefix_sample[1])
    return "\n".join(prefixes)


def test_LreConceptTrainer_train_all(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    template = "{} is located in the country of"
    japan_cities = [
        "Tokyo",
        "Osaka",
        "Nagoya",
        "Hiroshima",
        "Yokohama",
        "Kyoto",
        "Nagasaki",
        "Kobe",
        "Kitashima",
        "Kyushu",
    ]
    china_cities = [
        "Beijing",
        "Shanghai",
        "Nanjing",
        "Hangzhou",
        "Peking",
        "Qingdao",
        "Chongqing",
        "Changsha",
        "Wuhan",
        "Chengdu",
    ]
    samples: list[tuple[str, str]] = []
    for city in japan_cities:
        samples.append((city, "Japan"))
    for city in china_cities:
        samples.append((city, "China"))
    samples = stable_shuffle(samples)
    prompts = []
    for city, country in samples:
        prefix = fsl_prefix((city, country), samples, template)
        prompts.append(
            create_prompt(city, country, text=prefix + "\n" + template.format(city))
        )

    trainer = Trainer(
        model,
        tokenizer,
        layer_matcher="transformer.h.{num}",
    )

    concepts = trainer.train_relation_concepts(
        relation="located_in_country",
        subject_layer=8,
        object_layer=10,
        prompts=prompts,
        inv_lre_rank=50,
    )

    assert len(concepts) == 2
    for concept in concepts:
        assert concept.layer == 8
        assert concept.vector.shape == (768,)
        assert concept.vector.norm() == pytest.approx(1.0)

    # # evaluating on the train set should score highly
    # evaluator = Evaluator(
    #     model,
    #     tokenizer,
    #     layer_matcher="transformer.h.{num}",
    #     dataset=dataset,
    #     prompt_validator=trainer.prompt_validator,
    # )
    # accuracy_results = evaluator.evaluate_accuracy(concepts, verbose=False)
    # assert accuracy_results["located_in_country"].accuracy > 0.9
