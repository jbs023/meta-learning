TODO: JUST MAKE THIS A FUNCTION PACKAGE

# Meta-Learning

A reimplementation of some popular approaches.

1. [Siamese Neural Networks](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)
2. [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf)
3. [Prototypical Networks](https://arxiv.org/pdf/1703.05175.pdf)
4. [WIP: MANN](https://arxiv.org/pdf/1605.06065.pdf)
5. [WIP: Model Agnostic Meta-Learning](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)

This repo is an exercise for my own education and heavily relies on other people's code. I try to cite the necessary sources when appropriate. Additionally, there are many things that are redudant and un-optimized. One day I hope to go through and make everything pretty, but you know how grad school is.

A lot of this is slow to run, but simple to read (hopefully).

Methods to implement/test:

1. MAML
2. Prob MAML
3. DKT
4. Conditional Neural Processes
5. Other cool one's probably

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

