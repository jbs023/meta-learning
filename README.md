# Meta-Learning

A reimplementation of some popular approaches
1. [Siamese Neural Networks](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)
2. [MANN](https://arxiv.org/pdf/1605.06065.pdf)
3. [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf)
4. [Prototypical Networks](https://arxiv.org/pdf/1703.05175.pdf)
5. [WIP: Model Agnostic Meta-Learning](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)

Note: Implementations are all a little different, will standardize them one day

## Omniglot dataset
Right now, each model has a different data loading scheme. The Siamese Network model needs the data to be manually downloaded.

```
mkdir data

git clone https://github.com/brendenlake/omniglot.git
cd omniglot/python

cp images_background.zip ../../data
cp images_evaluation.zip ../../data
cd ../../data

unzip images_background.zip
unzip images_evaluation.zip

cd ../
rm -r omniglot/
```

## Usage
The following is the general procedure for training each model:

```
cd src/<model_type_dir>
python train.py

tensorboard --logdir=run/<model_type_name>

```

## Requirements
- pytorch
- tensorboard
- torchmeta
- python3.8

