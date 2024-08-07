Advanced usage
==============

Bulk editing
''''''''''''

Edits can be performed in batches to make better use of GPU resources using ``editor.swap_subject_concepts_and_predict_greedy_bulk()`` as below:

.. code:: python

    from linear_relational import CausalEditor, ConceptSwapAndPredictGreedyRequest

    concepts = trainer.train_relation_concepts(...)

    editor = CausalEditor(model, tokenizer, concepts=concepts)

    swap_requests = [
    ConceptSwapAndPredictGreedyRequest(
        text="Shanghai is located in the country of",
        subject="Shanghai",
        remove_concept="located in country: China",
        add_concept="located in country: France",
        predict_num_tokens=1,
    ),
    ConceptSwapAndPredictGreedyRequest(
        text="Berlin is located in the country of",
        subject="Berlin",
        remove_concept="located in country: Germany",
        add_concept="located in country: Japan",
        predict_num_tokens=1,
    ),
    ]
    edited_answers = editor.swap_subject_concepts_and_predict_greedy_bulk(
        requests=swap_requests,
        edit_single_layer=False,
        magnitude_multiplier=0.1,
        batch_size=4,
    )
    print(edited_answers) # [" France", " Japan"]


Bulk concept matching
'''''''''''''''''''''

We can perform concept matches in batches to better utilize GPU resources using ``matcher.query_bulk()`` as below:

.. code:: python

    from linear_relational import ConceptMatcher, ConceptMatchQuery

    concepts = trainer.train_relation_concepts(...)

    matcher = ConceptMatcher(model, tokenizer, concepts=concepts)

    match_queries = [
        ConceptMatchQuery("Beijng is a northern city", subject="Beijing"),
        ConceptMatchQuery("I sawi him in Marseille", subject="Marseille"),
    ]
    matches = matcher.query_bulk(match_queries, batch_size=4)

    print(matches[0].best_match.concept) # located in country: China
    print(matches[1].best_match.concept) # located in country: France


Customizing LRC training
''''''''''''''''''''''''

The base ``trainer.train_relation_concepts()`` function is a convenience wrapper which trains a LRE,
performs a low-rank inverse of the LRE, and uses the inverted LRE to generate concepts. If you want to customize
this process, you can generate a LRE using ``trainer.train_lre()``, followed by inverting the LRE with ``lre.invert()``,
and finally training concepts from the inverted LRE with ``trainer.train_relation_concepts_from_inv_lre()``. This process
is shown below:

.. code:: python

    from linear_relational import Trainer

    trainer = Trainer(model, tokenizer)
    prompts = [...]

    lre = trainer.train_lre(...)
    inv_lre = lre.invert(rank=200)

    concepts = trainer.train_relation_concepts_from_inv_lre(
        inv_lre=inv_lre,
        prompts=prompts,
    )


It's also possible to pass a lambda function as the ``inv_lre`` param to allow using a different inverted LRE
for each object. This lambda takes the object as a string and returns the inverted LRE for that object. However,
if you use this approach, you must also pass in ``relation``, ``object_aggregation`` and ``object_layer``, as these
cannot be inferred from the inverted LRE when passed as a function.

This is shown below:

.. code:: python

    from linear_relational import Trainer

    trainer = Trainer(model, tokenizer)
    prompts = [...]

    lre1 = trainer.train_lre(...)
    inv_lre1 = lre.invert(rank=200)

    lre2 = trainer.train_lre(...)
    inv_lre2 = lre.invert(rank=200)

    def inv_lre_fn(object_name):
        return inv_lre1 if object_name == "Paris" else inv_lre2

    concepts = trainer.train_relation_concepts_from_inv_lre(
        inv_lre=inv_lre_fn,
        prompts=prompts,
        relation="located_in_country",
        object_aggregation="mean",
        object_layer=20,
    )


Custom objects in prompts
'''''''''''''''''''''''''

By default, when you create a ``Prompt``, the answer to the prompt is assumed to be the object 
corresponding to a LRC. For instance, in the prompt ``Prompt("Paris is located in", "France", subject="Paris")``,
the answer, "France", is assumed to be the object. However, if this is not the case, you can specify the object
explicitly using the ``object_name`` parameter as below:

.. code:: python

    from linear_relational import Prompt

    prompt1 = Prompt(
        text="PARIS IS LOCATED IN",
        answer="FRANCE",
        subject="PARIS",
        object_name="france",
    )
    prompt2 = Prompt(
        text="Paris is located in",
        answer="France",
        subject="Paris",
        object_name="france",
    )


Skipping prompt validation
''''''''''''''''''''''''''

By default, the ``Trainer`` will validate that for every prompt passed in, that the model answers the prompt correctly,
and will filter out any prompts where this is not the case.
If you want to skip this validation, you can pass ``validate_prompts=False`` to all methods on the trainer
like ``Trainer.train_relation_concepts(prompts, validate_prompts=False)``.


Multi-token object aggregation
''''''''''''''''''''''''''''''

If a prompt has an answer which is multiple tokens, by default the ``Trainer`` will use the mean activation of 
the tokens in the answer when training a LRE. An example of a prompt with a multi-token answer is "The CEO of Microsoft is Bill Gates",
where the object, "Bill Gates", has two tokens. Alternatively, you can use just the first token of the object by
passing ``object_aggregation="first_token"`` when training a LRE. For instance, you can run the following:

.. code:: python

    lre = trainer.train_lre(
        prompts=prompts,
        object_aggregation="first_token",
    )

If the answer is a single token, "mean" and "first_token" are equivalent. 


Custom layer selection
''''''''''''''''''''''

By default, the library will try to guess which layers corresponding to hidden activations in the model,
and will use these layers for reading activations and training LREs. If the layers the library guesses are not
correct, or if you want to use different layers to extract activations and train LREs, you can pass in a 
custom ``layer_matcher`` to the ``Trainer``, ``CausalEditor``, and ``ConceptMatcher`` when creating these
objects.

A ``layer_matcher`` is typically A string, and must include the substring ``"{num}"`` which will be replaced
with the layer number to select a layer in the model. For instance, for GPT models, the matcher for
hidden layers is ``"transformer.h.{num}"``. You can find a list of all layers in a model by calling
``model.named_modules()``.

For most cases, using a string is sufficient, but if you want to customize the layer matcher further
you can pass in a function to ``layer_matcher`` which takes in the layer number as an int and 
returns the layer in the model as a string. For instance, for GPT models, this could be provided as
``lambda num: f"transformer.h.{num}"``.

