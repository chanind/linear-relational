Linear-Relational
=================================================

A library for working with Linear Relational Embeddings (LREs) and Linear Relational Concepts (LRCs) for LLMs in PyTorch

.. image:: https://img.shields.io/pypi/v/linear-relational.svg?color=blue
   :target: https://pypi.org/project/linear-relational
   :alt: PyPI

.. image:: https://img.shields.io/github/actions/workflow/status/chanind/linear-relational/ci.yaml?branch=main
   :target: https://github.com/chanind/linear-relational
   :alt: Build Status


Installation
------------
Linear-Relational releases are hosted on `PyPI`_, and can be installed using `pip` as below:

.. code-block:: bash

   pip install linear-relational

This library assumes you're working with PyTorch and Huggingface Transformers.

LREs and LRCs
-------------

This library provides utilities and PyTorch modules for working with LREs and LRCs. LREs estimate the relation between a subject and object in a transformer language model (LM) as a linear map.

This library assumes you're working with sentences with a subject, relation, and object. For instance, in the sentence: "Lyon is located in the country of France" would have the subject "Lyon", relation "located in country", and object "France". A LRE models a relation like "located in country" as a linear map consisting of a weight matrix :math:`W` and a bias term :math:`b`, so a LRE would map from the activations of the subject (Lyon) at layer :math:`l_s` to the activations of the object (France) at layer :math:`l_o`. So:

.. math::
   LRE(s) = W s + b

LREs can be inverted using a low-rank inverse, shown as :math:`LRE^{\dagger}`, to estimate :math:`s` from :math:`o`:

.. math::
   LRE^{\dagger}(o) = W^{\dagger}(o - b)

Linear Relational Concepts (LRCs) represent a concept :math:`(r, o)` as a direction vector $v$ on subject tokens, and can act like a simple linear classifier. For instance, while a LRE can represent the relation "located in country", we could learn a LRC for "located in the country: France", "located in country: Germany", "located in country: China", etc... This is just the result from passing in an object activation into the inverse LRE equation above.

.. math::
   LRC(o) = W^{\dagger}(o - b)

For more information on LREs and LRCs, check out `these <https://arxiv.org/abs/2308.09124>`_ `papers <https://arxiv.org/abs/2311.08968>`_.


.. toctree::
   :maxdepth: 2

   basic_usage
   advanced_usage
   about

.. toctree::
   :caption: Project Links

   GitHub <https://github.com/chanind/linear-relational>
   PyPI <https://pypi.org/project/linear-relational>

.. _PyPI: https://pypi.org/project/linear-relational/
