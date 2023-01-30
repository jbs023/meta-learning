TODO: JUST MAKE THIS A FUNCTION PACKAGE

# Meta-Learning

A reimplementation of some popular approaches.

1. [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf)
2. [Prototypical Networks](https://arxiv.org/pdf/1703.05175.pdf)
3. [Model Agnostic Meta-Learning](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)

This repo is an exercise for my own education and heavily relies on other people's code. I try to cite the necessary sources when appropriate. Additionally, there are many things that are redudant and un-optimized. One day I hope to go through and make everything pretty, but you know how grad school is.

A lot of this is slow to run, but simple to read (hopefully).

Methods to implement/test:

1. Prob MAML
2. DKT
3. Conditional Neural Processes
4. Other cool one's probably

## Usage
The following is the general procedure for training each model:

```
pip install -e .

cd meta_learn/<model_type_dir>
python train.py

tensorboard --logdir=run/<model_type_name>

```

## Requirements
- pytorch
- tensorboard
- python3.8

