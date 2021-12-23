import numpy as np
import os
import random

from numpy.core.defchararray import count
import torch


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """

    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, data_folder="./omniglot_resized", device = torch.device('cpu')):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)
            
            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the querry set.
                
            device: cuda.device: 
                Device to allocate tensors to.
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.img_size = (28, 28)

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]
        self.device = device

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: str
                train/val/test set to sample from
                
            batch_size: int:
                Size of batch of tasks to sample
                
        Returns:
            images: tensor
                A tensor of images of size [B, K+1, N, 784]
                where B is batch size, K is number of samples per class, 
                N is number of classes
                
            labels: tensor
                A tensor of images of size [B, K+1, N, N] 
                where B is batch size, K is number of samples per class, 
                N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        # SOLUTION:
        batch_image_array = np.zeros((batch_size, self.num_samples_per_class+1, self.num_classes, 784))
        batch_label_array = np.zeros((batch_size, self.num_samples_per_class+1, self.num_classes, self.num_classes))
        for b in range(batch_size):
            #Random sample alphabets for characters
            random_classes = torch.randint(0, len(folders), (self.num_classes,))

            #Randomly sample examples of each random character
            image_labels = get_images(
                [folders[i] for i in random_classes], 
                np.arange(0, self.num_classes), 
                nb_samples=self.num_samples_per_class+1
            )

            #Read images and flatten
            label_dict = dict.fromkeys(np.arange(0, self.num_classes), 0)
            image_array = np.zeros((self.num_samples_per_class+1, self.num_classes, 784))
            label_array = np.zeros((self.num_samples_per_class+1, self.num_classes, self.num_classes))

            for label, image in image_labels:
                index = label_dict[label]
                new_label = np.zeros((self.num_classes,))
                new_label[label] = 1

                label_array[index][label] = new_label
                image_array[index][label] = np.array(image_file_to_array(image, 784))
                label_dict[label] += 1

            #Shuffle the query set (i.e, the last element of both lists)
            shuffler = np.random.permutation(len(label_array[-1]))
            label_array[-1] = label_array[-1][shuffler]
            image_array[-1] = image_array[-1][shuffler]
            
            batch_image_array[b] = image_array
            batch_label_array[b] = label_array

        return torch.tensor(batch_image_array), torch.tensor(batch_label_array)

        

            
