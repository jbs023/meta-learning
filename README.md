# Siamese Networks for Few-Shot Learning

A reimplementation of the [original paper](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf).

## Omniglot dataset
- download dataset
```
git clone https://github.com/brendenlake/omniglot.git
cd omniglot/python
unzip images_evaluation.zip
unzip images_background.zip

cd ../..
# setup directory for saving models
mkdir cnn_models

##Requirements
- pytorch
- python3.6+
