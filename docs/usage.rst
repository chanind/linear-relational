Usage
=====
This library assumes you're using PyTorch with a decoder-only generative language
model (e.g., GPT, LLaMa, etc...), and a tokenizer from Huggingface.

Training a LRE
''''''''''''''

To train a LRE for a relation, first collect prompts which elicit the relation.
We provide a ``Prompt`` class to represent this data, and a ``Trainer`` class to make training
a LRE easy. Below, we train a LRE to represent the "located in country" relation.

.. code-block:: python

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from linear_relational import Prompt, Trainer

    # We load a generative LM from huggingface. The LMHead must be included.
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Prompts consist of text, an answer, and subject.
    # The subject must appear in the text. The answer
    # is what the model should respond with, and corresponds to the "object"
    prompts = [
        Prompt("Paris is located in the country of", "France", subject="Paris"),
        Prompt("Shanghai is located in the country of", "China", subject="Shanghai"),
        Prompt("Kyoto is located in the country of", "Japan", subject="Kyoto"),
        Prompt("San Jose is located in the country of", "Costa Rica", subject="San Jose"),
    ]

    trainer = Trainer(model, tokenizer)

    lre = trainer.train_lre(
        relation="located in country",
        subject_layer=8, # subject layer must be before the object layer
        object_layer=10,
        prompts=prompts,
    )

Working with a LRE
''''''''''''''''''

A LRE is a PyTorch module, so once a LRE is trained, we can use it to predict object activations from subject activations:

.. code-block:: python

    object_acts_estimate = lre(subject_acts)

We can also create a low-rank estimate of the LRE:

.. code-block:: python

    low_rank_lre = lre.to_low_rank(50)
    low_rank_obj_acts_estimate = low_rank_lre(subject_acts)

Finally we can invert the LRE:

.. code-block:: python

    inv_lre = lre.invert(rank=50)
    subject_acts_estimate = inv_lre(object_acts)

Training LRCs for a relation
''''''''''''''''''''''''''''

The ``Trainer`` can also create LRCs for a relation. Internally, this first create a LRE, inverts it,
then generates LRCs from each object in the relation. Objects refer to the answers in the prompts,
e.g. in the example above, "France" is an object, "Japan" is an object, etc...

.. code-block:: python

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from linear_relational import Prompt, Trainer

    # We load a generative LM from huggingface. The LMHead must be included.
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Prompts consist of text, an answer, and subject.
    # The subject must appear in the text. The answer
    # is what the model should respond with, and corresponds to the "object"
    prompts = [
        Prompt("Paris is located in the country of", "France", subject="Paris"),
        Prompt("Shanghai is located in the country of", "China", subject="Shanghai"),
        Prompt("Kyoto is located in the country of", "Japan", subject="Kyoto"),
        Prompt("San Jose is located in the country of", "Costa Rica", subject="San Jose"),
    ]

    trainer = Trainer(model, tokenizer)

    concepts = trainer.train_relation_concepts(
        relation="located in country",
        subject_layer=8,
        object_layer=10,
        prompts=prompts,
        max_lre_training_samples=10,
        inv_lre_rank=50,
    )

Causal editing
''''''''''''''

Once we have LRCs trained, we can use them to perform causal edits while the model is running.
For instance, we can perform a causal edit to make the model output that
"Shanghai is located in the country of France" by subtracting the
"located in country: China" concept from "Shanghai" and adding the
"located in country: France" concept. We can use the ``CausalEditor`` class to perform these edits.

.. code-block:: python

    from linear_relational import CausalEditor

    concepts = trainer.train_relation_concepts(...)

    editor = CausalEditor(model, tokenizer, concepts=concepts)

    edited_answer = editor.swap_subject_concepts_and_predict_greedy(
        text="Shanghai is located in the country of",
        subject="Shanghai",
        remove_concept="located in country: China",
        add_concept="located in country: France",
        edit_single_layer=8,
        magnitude_multiplier=1.0,
        predict_num_tokens=1,
    )
    print(edited_answer) # " France"


Single-layer vs multi-layer edits
'''''''''''''''''''''''''''''''''

Above we performed a single-layer edit, only modifying subject activations at layer 8.
However, we may want to perform an edit at all subject layers at the same time instead.
To do this, we can pass ``edit_single_layer=False`` to ``editor.swap_subject_concepts_and_predict_greedy()``.
We should also reduce the ``magnitude_multiplier`` since now we're going to make the edit at every layer, if we use 
too large of a multiplier we'll drown out the rest of the activations in the model. The ``magnitude_multiplier`` is a
hyperparameter that requires tuning depending on the model being edited.

.. code:: python

    from linear_relational import CausalEditor

    concepts = trainer.train_relation_concepts(...)

    editor = CausalEditor(model, tokenizer, concepts=concepts)

    edited_answer = editor.swap_subject_concepts_and_predict_greedy(
        text="Shanghai is located in the country of",
        subject="Shanghai",
        remove_concept="located in country: China",
        add_concept="located in country: France",
        edit_single_layer=False,
        magnitude_multiplier=0.1,
        predict_num_tokens=1,
    )
    print(edited_answer) # " France"

Bulk editing
''''''''''''

Edits can be performed in batches to make better use of GPU resources using `editor.swap_subject_concepts_and_predict_greedy_bulk()` as below:

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

Concept matching
''''''''''''''''

We can use learned concepts (LRCs) to act like classifiers and match them against subject activations in sentences.
We can use the ``ConceptMatcher`` class to do this matching.

.. code:: python

    from linear_relational import ConceptMatcher

    concepts = trainer.train_relation_concepts(...)

    matcher = ConceptMatcher(model, tokenizer, concepts=concepts)

    match_info = matcher.query("Beijing is a northern city", subject="Beijing")

    print(match_info.best_match.name) # located in country: China
    print(match_info.betch_match.score) # 0.832

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

    print(matches[0].best_match.name) # located in country: China
    print(matches[1].best_match.name) # located in country: France
