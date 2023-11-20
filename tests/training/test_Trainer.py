import pytest
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from linear_relational.ConceptMatcher import ConceptMatcher, ConceptMatchQuery
from linear_relational.lib.util import stable_shuffle
from linear_relational.Prompt import Prompt
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


def prompts_from_samples(samples: list[tuple[str, str]], template: str) -> list[Prompt]:
    prompts = []
    for city, country in samples:
        prefix = fsl_prefix((city, country), samples, template)
        prompts.append(
            create_prompt(city, country, text=prefix + "\n" + template.format(city))
        )
    return prompts


def test_Trainer_train_relation_concepts(
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
    prompts = prompts_from_samples(samples, template)

    trainer = Trainer(model, tokenizer)

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

    # test the learned concepts match the correct city activations
    matcher = ConceptMatcher(model, tokenizer, concepts)
    japan_results = matcher.query_bulk(
        [ConceptMatchQuery(template.format(city), city) for city in japan_cities]
    )
    china_results = matcher.query_bulk(
        [ConceptMatchQuery(template.format(city), city) for city in china_cities]
    )
    for japan_result in japan_results:
        assert japan_result.best_match.concept == "located_in_country: Japan"
    for china_result in china_results:
        assert china_result.best_match.concept == "located_in_country: China"


def test_Trainer_train_relation_concepts_allows_overriding_concept_names(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    template = "{} is located in the country of"
    japan_cities = ["Tokyo", "Osaka", "Nagoya"]
    china_cities = ["Beijing", "Shanghai", "Nanjing"]
    samples: list[tuple[str, str]] = []
    for city in japan_cities:
        samples.append((city, "Japan"))
    for city in china_cities:
        samples.append((city, "China"))
    samples = stable_shuffle(samples)
    prompts = prompts_from_samples(samples, template)

    trainer = Trainer(model, tokenizer)

    concepts = trainer.train_relation_concepts(
        relation="located_in_country",
        name_concept_fn=lambda _relation, object: object,
        subject_layer=8,
        object_layer=10,
        prompts=prompts,
        max_lre_training_samples=2,
    )
    concept_names = [concept.name for concept in concepts]
    assert "Japan" in concept_names
    assert "China" in concept_names
