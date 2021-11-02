# Siamese Networks for Few-Shot Learning

A reimplementation of the [original paper](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf).

## Omniglot dataset
```
git clone https://github.com/brendenlake/omniglot.git
cd omniglot/python
unzip images_evaluation.zip
unzip images_background.zip
unzip images_background_small1.zip

cp -r image_evaluation ../../
cp -r images_background ../../
cp -r images_background_small1 ../../

rm -r omniglot/
```

## Requirements
- pytorch
- python3.6+

