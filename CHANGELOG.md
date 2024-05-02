# CHANGELOG



## v0.3.1 (2024-05-02)

### Fix

* fix: fixing bug in inverted LRE forward() ([`46510ae`](https://github.com/chanind/linear-relational/commit/46510ae2f5f44f95a4ca9defef3e4d74bb6e44e8))


## v0.3.0 (2024-05-02)

### Chore

* chore: adding codecov (#4) ([`d25bc48`](https://github.com/chanind/linear-relational/commit/d25bc4803fe7d4748ada69843db4c63036e97beb))

* chore: Update README.md ([`d35bc8b`](https://github.com/chanind/linear-relational/commit/d35bc8b540862429338fbddba809fe9c06099b48))

* chore: removing unused constant ([`5b60e5b`](https://github.com/chanind/linear-relational/commit/5b60e5b546f4a9a2cb2378d6d48dcab6c3643d40))

### Feature

* feat: allow setting max prompts in LRE training (#5) ([`2066999`](https://github.com/chanind/linear-relational/commit/2066999bf4d60ae63beb287dbb367c0de78006c0))


## v0.2.1 (2024-02-16)

### Fix

* fix: exporting ObjectAggregation and VectorAggregation from main module ([`975e215`](https://github.com/chanind/linear-relational/commit/975e215cc44f7d45f6c676ecd2efc73f5db9a546))


## v0.2.0 (2024-02-16)

### Chore

* chore: update setup-python action to use node 20 ([`18051cc`](https://github.com/chanind/linear-relational/commit/18051cc2c1615abb7af32db0aa80c7b0ffe81fdb))

### Feature

* feat: allow passing a callable as inv_lre to train_relation_concepts_from_inv_lre() (#2) ([`12fa48e`](https://github.com/chanind/linear-relational/commit/12fa48e77e7ee7901420557b9eea0055f02381c8))


## v0.1.4 (2024-02-15)

### Chore

* chore: fixing typing for newer torch types ([`3f668d9`](https://github.com/chanind/linear-relational/commit/3f668d964d0ea7a06742c4286f93250b1fe01abc))

* chore: fix typo in  ci.yaml ([`2492912`](https://github.com/chanind/linear-relational/commit/249291238aa0665cc5c7e1280c21c0302f0e0b0c))

* chore: fixing typo in docs ([`a6ddd0b`](https://github.com/chanind/linear-relational/commit/a6ddd0b7a1442f770ab3d2e0e9c8c76b3664aacd))

### Fix

* fix: adding py.typed marker to declare type inference ([`5249dd3`](https://github.com/chanind/linear-relational/commit/5249dd37f347457afcb017527a5014c2729b936f))


## v0.1.3 (2023-11-23)

### Fix

* fix: match dtype of activations during Concept.forward ([`3a06459`](https://github.com/chanind/linear-relational/commit/3a064595c314d5b9ca1944e9eb4541a041128262))


## v0.1.2 (2023-11-23)

### Chore

* chore: Update README.md ([`9e85cf0`](https://github.com/chanind/linear-relational/commit/9e85cf040d658b2df2f971f32fb988f3f3cc3ea9))

* chore: improving docs ([`c1fb709`](https://github.com/chanind/linear-relational/commit/c1fb7096314674de45237d879230180559a37408))

* chore: improving docs ([`ab4748e`](https://github.com/chanind/linear-relational/commit/ab4748e01d8f2c08c173491c7f9993312e826b89))

* chore: updating docs with advanced usage ([`7439e71`](https://github.com/chanind/linear-relational/commit/7439e71bdbf0379deadeb983b8f0116bcf049c92))

* chore: Add docs link to README.md ([`b915827`](https://github.com/chanind/linear-relational/commit/b915827b06f3442eaff73b49e382e4e93da124cf))

* chore: typo in docs ([`b4f547a`](https://github.com/chanind/linear-relational/commit/b4f547a329af3a4fa2b4f91212c72d695e56b019))

### Fix

* fix: set device when checking if answers match ([`008c4b6`](https://github.com/chanind/linear-relational/commit/008c4b61fc707308e7cb4beafe8f4c065051eb48))

### Unknown

* Update README.md ([`33e7ebd`](https://github.com/chanind/linear-relational/commit/33e7ebd426b4e61dc9b7564057eb42d92ffc26e7))


## v0.1.1 (2023-11-23)

### Chore

* chore:  add Sphinx docs (#1)

* adding a basic docs page and validating in ci

* adding deploy pages action ([`4088d61`](https://github.com/chanind/linear-relational/commit/4088d61da5479fe7578511ed2afd3059cbd2341a))

* chore: Adding badges to README.md ([`341b6cd`](https://github.com/chanind/linear-relational/commit/341b6cdd0c522e7492f3b49e1430a0e1fed33375))

### Fix

* fix: docs ([`ed8a203`](https://github.com/chanind/linear-relational/commit/ed8a203d25ecb8e55f11d24cf2f687ad0477fed4))


## v0.1.0 (2023-11-22)

### Feature

* feat: first version ([`6685bbf`](https://github.com/chanind/linear-relational/commit/6685bbf0a1257e2f027db78468b3f93e175f7bf1))


## v0.0.0 (2023-11-22)

### Unknown

* moving protobuf to dev deps ([`a6fdcec`](https://github.com/chanind/linear-relational/commit/a6fdcec46bd5a7744b8de6a6e2e0adc435bc7860))

* moving sentencepiece to dev reqs ([`03e38f5`](https://github.com/chanind/linear-relational/commit/03e38f5fa66d375972668ed3ed6cd812002ed298))

* fixing python version ([`bf451de`](https://github.com/chanind/linear-relational/commit/bf451dea4f71282718a14c55ad7c99327d71c916))

* synchronizing version ([`0bc4130`](https://github.com/chanind/linear-relational/commit/0bc4130bc5df8b141ac6eb6d98e51aa8ba777eee))

* setting up deployment ([`369660c`](https://github.com/chanind/linear-relational/commit/369660cf197aa08f82568b3bb2c61712f869b571))

* minor change to readme ([`9dc7a81`](https://github.com/chanind/linear-relational/commit/9dc7a813a236ae526dc20404ce40b8e29fdd4af2))

* removing name of paper from links ([`8bd04f0`](https://github.com/chanind/linear-relational/commit/8bd04f05e230910ecf17ab7988888093c7368ec9))

* renaming batch params to be more specific ([`b6c70b4`](https://github.com/chanind/linear-relational/commit/b6c70b4a67ad7562dfec1eaa4d98675f1a7b9e69))

* removing caching as it throws errors ([`53649e3`](https://github.com/chanind/linear-relational/commit/53649e35c5831962708522f00b15cf84a2f075a8))

* updating dependency cache ([`4ff0098`](https://github.com/chanind/linear-relational/commit/4ff00980afa5ea2299a8b2bf9e9b053525ef0d50))

* adding back dataclasses-json dep ([`30eef6a`](https://github.com/chanind/linear-relational/commit/30eef6a2dece3cdcf610ec9e30e98a08c8db4cac))

* removing poetry.lock ([`5237bfb`](https://github.com/chanind/linear-relational/commit/5237bfb10e1f8db3afbb9144b826051f2b0a5d79))

* removing unused dependency ([`305ed04`](https://github.com/chanind/linear-relational/commit/305ed044e9f3c842efb7e47eef9071a3c128ba51))

* Update README.md ([`6839ca0`](https://github.com/chanind/linear-relational/commit/6839ca044beceac53caba049df6566efc6a04205))

* tweaking README and adding base module exports ([`902ee43`](https://github.com/chanind/linear-relational/commit/902ee43c4d66a79d05a55a87d939c8ae955aa7cd))

* updating README ([`1bd8e31`](https://github.com/chanind/linear-relational/commit/1bd8e31bfd9adc319a7972fdf8286b1218c38dce))

* improving tests for trainer ([`c9d0f69`](https://github.com/chanind/linear-relational/commit/c9d0f694b496704ffe856e93b631eeb7daeab22d))

* default to guessing layer matcher if not provided ([`9c52625`](https://github.com/chanind/linear-relational/commit/9c5262572284ac3c0160b5427a13da79261b4108))

* adding a helper to guess the hidden layers of the model ([`c1c01d6`](https://github.com/chanind/linear-relational/commit/c1c01d6a35ef1b1eb6292a2da8a99d011ca2b495))

* adding a trainer class ([`fcf673d`](https://github.com/chanind/linear-relational/commit/fcf673d1b43780f0acef4e6f7b072322b76a5f88))

* adding citation info ([`787b6de`](https://github.com/chanind/linear-relational/commit/787b6dec878c798f37634d317a1483f6cae32b64))

* adding initial libs and classes ([`2691e8f`](https://github.com/chanind/linear-relational/commit/2691e8ff57dd4d997712668733473c8374510e8b))

* Initial commit ([`2ccf587`](https://github.com/chanind/linear-relational/commit/2ccf58722ca52a967653a8acc877407ce1ad62c0))
