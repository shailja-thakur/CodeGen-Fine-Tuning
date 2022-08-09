# CodeGen-Fine-Tuning

## This repository provides steps for fine-tuning CodeGen language models on specific programming languages.

CodeGen is a suite of code based language models by SalesForce. Model sizes vary with respect to their training corpus, and model parameters. Models are named as per the convention codegen-{model-size}-{data}.

`model-size` has 4 options: `350M`, `2B`, `6B`, `16B`, which represent the number of parameters in each model.

`data` has 3 options: `nl`, `multi`, `mono`.

* `nl` models are randomly initialized and trained on [The Pile](https://github.com/EleutherAI/the-pile), a 825.18 GB English text corpus.
* `multi` models are initialized from `nl` models and then trained on a corpus with code data consisting of multiple programming languages.
* `mono` models are initialized from `multi` models and then trained on a corpus with Python code data.

A detailed description of the models isas follows:

## CodeGen models

| model name | data | model-size|
| ------ | ----------- |--------|
| codegen-350M-nl   | nl |350M|
| codegen-350M-multi  | multi | 350M |
| codegen-350M-mono    | mono| 350M|
|codegen-2B-nl       | nl| 2B|
|codegen-2B-multi      |multi| 2B|
|codegen-2B-mono       |mono| 2B|
|codegen-6B-nl       |nl|6B|
|codegen-6B-multi      |multi|6B|
|codegen-6B-mono       |mono|6B|
|codegen-16B-nl      |nl|16B|
|codegen-16B-multi       |multi|16B|
|codegen-16B-mono      |mono|16B|



Following is a description of the machine where the fine-tuning succeeded

