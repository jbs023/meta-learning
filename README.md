# Meta-Learning

A reimplementation of some popular approaches
1. [Siamese Neural Networks](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)
2. [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf)

## Omniglot dataset
Siamese and Matching code examples use different dataloaders. The Siamese dataloader requires you to download the files upfront. The Matching Network code does not.

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
- torchmeta
- python3.6+

