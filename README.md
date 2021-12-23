# Meta-Learning

A reimplementation of some popular approaches
1. [Siamese Neural Networks](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)
2. [MANN](https://arxiv.org/pdf/1605.06065.pdf)
3. [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf)
4. [Prototypical Networks](https://arxiv.org/pdf/1703.05175.pdf)
5. [WIP: Model Agnostic Meta-Learning](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)

Note: Implementations are all a little different, will standardize them one day

## Omniglot dataset
Each module uses a different dataloader. The Siamese dataloader requires you to download the files upfront. Everything else should handle the data management automatically.

```
git clone https://github.com/brendenlake/omniglot.git
cd omniglot/python
unzip images_evaluation.zip
unzip images_background.zip
unzip images_background_small1.zip

cp -r image_evaluation ../../data/
cp -r images_background ../../data/
cp -r images_background_small1 ../../data/
```

## Usage
The following is the general procedure for training and testing each model:

```
cd src/<model_type_dir>

#Train
python train.py -t -s <saved_model_file_suffix>

#Test
python train.py -s <saved_model_file_suffix>
```

The matching network implementation will automatically download the data you need, but the Siamese Network implementation requires you to following the procedure in the preceding section.

## Requirements
- pytorch
- torchmeta
- python3.6+

