#Reference: https://github.com/tristandeleu/pytorch-meta
#This code is a trimmed down version of data utils from pytorch-meta package
#I am not using the package because it seems to be no longer supported.
#
#Additionally, installing caused conflicts and code decomposition and
#re-use isn't the worst exercise


from collections import OrderedDict
import torch
import os
import json
import glob
import h5py
import random
import sys
import warnings
import shutil
import zipfile
import tarfile
import pickle
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image, ImageOps

from copy import deepcopy
from itertools import combinations
from ordered_set import OrderedSet
from collections import defaultdict
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import Dataset as Dataset_

from google_drive_downloader import GoogleDriveDownloader as gdd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

from torchvision.transforms import Compose
from torchvision.datasets.utils import list_dir, download_url
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as TorchDataset

#TODO: Reduce all of this to bare minimum for understandability reasons
#which may require a bit of re-implementation
def no_collate(batch):
    return batch

def get_asset_path(*args):
    basedir = os.path.dirname(__file__)
    return os.path.join(basedir, 'assets', *args)

def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    print(filename)
    if not os.path.isfile(filename):
        raise IOError('{} not found'.format(filename))

    if dtype is None:
        _, dtype = os.path.splitext(filename)
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    return data

def apply_wrapper(wrapper, task_or_dataset=None):
    if task_or_dataset is None:
        return wrapper

    if isinstance(task_or_dataset, Task):
        return wrapper(task_or_dataset)
    elif isinstance(task_or_dataset, MetaDataset):
        if task_or_dataset.dataset_transform is None:
            dataset_transform = wrapper
        else:
            dataset_transform = Compose([
                task_or_dataset.dataset_transform, wrapper])
        task_or_dataset.dataset_transform = dataset_transform
        return task_or_dataset
    else:
        raise NotImplementedError()
    
def wrap_transform(transform, fn, transform_type=None):
    if (transform_type is None) or isinstance(transform, transform_type):
        return fn(transform)
    elif isinstance(transform, Compose):
        return Compose([wrap_transform(subtransform, fn, transform_type)
            for subtransform in transform.transforms])
    else:
        return transform
    
def ClassSplitter(task=None, *args, **kwargs):
    return apply_wrapper(ClassSplitter_(*args, **kwargs), task)

class Splitter(object):
    def __init__(self, splits, random_state_seed):
        self.splits = splits
        self.random_state_seed = random_state_seed
        self.seed(random_state_seed)

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)

    def get_indices(self, task):
        if isinstance(task, ConcatTask):
            indices = self.get_indices_concattask(task)
        elif isinstance(task, Task):
            indices = self.get_indices_task(task)
        else:
            raise ValueError('The task must be of type `ConcatTask` or `Task`, '
                'Got type `{0}`.'.format(type(task)))
        return indices

    def get_indices_task(self, task):
        raise NotImplementedError('Method `get_indices_task` must be '
            'implemented in classes inherited from `Splitter`.')

    def get_indices_concattask(self, task):
        raise NotImplementedError('Method `get_indices_concattask` must be '
            'implemented in classes inherited from `Splitter`.')

    def _get_class_indices(self, task):
        class_indices = defaultdict(list)
        if task.num_classes is None: # Regression task
            class_indices['regression'] = range(len(task))
        else:
            for index in range(len(task)):
                sample = task[index]
                if (not isinstance(sample, tuple)) or (len(sample) < 2):
                    raise ValueError('In order to split the dataset in train/'
                        'test splits, `Splitter` must access the targets. Each '
                        'sample from a task must be a tuple with at least 2 '
                        'elements, with the last one being the target.')
                class_indices[sample[-1]].append(index)

            if len(class_indices) != task.num_classes:
                raise ValueError('The number of classes detected in `Splitter` '
                    '({0}) is different from the property `num_classes` ({1}) '
                    'in task `{2}`.'.format(len(class_indices),
                    task.num_classes, task))

        return class_indices

    def __call__(self, task):
        indices = self.get_indices(task)
        return OrderedDict([(split, SubsetTask(task, indices[split]))
            for split in self.splits])

    def __len__(self):
        return len(self.splits)


class ClassSplitter_(Splitter):
    def __init__(self, shuffle=True, num_samples_per_class=None,
                 num_train_per_class=None, num_test_per_class=None,
                 num_support_per_class=None, num_query_per_class=None,
                 random_state_seed=0):
        self.shuffle = shuffle

        if num_samples_per_class is None:
            num_samples_per_class = OrderedDict()
            if num_train_per_class is not None:
                num_samples_per_class['train'] = num_train_per_class
            elif num_support_per_class is not None:
                num_samples_per_class['support'] = num_support_per_class
            if num_test_per_class is not None:
                num_samples_per_class['test'] = num_test_per_class
            elif num_query_per_class is not None:
                num_samples_per_class['query'] = num_query_per_class
        assert len(num_samples_per_class) > 0

        self._min_samples_per_class = sum(num_samples_per_class.values())
        super(ClassSplitter_, self).__init__(num_samples_per_class, random_state_seed)

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])

        for i, (name, class_indices) in enumerate(all_class_indices.items()):
            num_samples = len(class_indices)
            if num_samples < self._min_samples_per_class:
                raise ValueError('The number of samples for class `{0}` ({1}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({2}).'.format(name,
                    num_samples, self._min_samples_per_class))

            if self.shuffle:
                seed = (hash(task) + i + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)
            else:
                dataset_indices = np.arange(num_samples)

            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += num_split

        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0

        for dataset in task.datasets:
            num_samples = len(dataset)
            if num_samples < self._min_samples_per_class:
                raise ValueError('The number of samples for one class ({0}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({1}).'.format(num_samples,
                    self._min_samples_per_class))

            if self.shuffle:
                seed = (hash(task) + hash(dataset) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)
            else:
                dataset_indices = np.arange(num_samples)

            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend(split_indices + cum_size)
                ptr += num_split
            cum_size += num_samples

        return indices


class Dataset(Dataset_):
    def __init__(self, index, transform=None, target_transform=None):
        self.index = index
        self.transform = transform
        self.target_transform = target_transform

    def target_transform_append(self, transform):
        if transform is None:
            return
        if self.target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = Compose([self.target_transform, transform])

    def __hash__(self):
        return hash(self.index)

class Task(Dataset):
    """Base class for a classification task.

    Parameters
    ----------
    num_classes : int
        Number of classes for the classification task.
    """
    def __init__(self, index, num_classes,
                 transform=None, target_transform=None):
        super(Task, self).__init__(index, transform=transform,
                                   target_transform=target_transform)
        self.num_classes = num_classes


class ConcatTask(Task, ConcatDataset):
    def __init__(self, datasets, num_classes, target_transform=None):
        index = tuple(task.index for task in datasets)
        Task.__init__(self, index, num_classes)
        ConcatDataset.__init__(self, datasets)
        for task in self.datasets:
            task.target_transform_append(target_transform)

    def __getitem__(self, index):
        return ConcatDataset.__getitem__(self, index)
    
class SubsetTask(Task, Subset):
    def __init__(self, dataset, indices, num_classes=None,
                 target_transform=None):
        if num_classes is None:
            num_classes = dataset.num_classes
        Task.__init__(self, dataset.index, num_classes)
        Subset.__init__(self, dataset, indices)
        self.dataset.target_transform_append(target_transform)

    def __getitem__(self, index):
        return Subset.__getitem__(self, index)

    def __hash__(self):
        return hash((self.index, tuple(self.indices)))

class TargetTransform(object):
    def __call__(self, target):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)


class DefaultTargetTransform(TargetTransform):
    def __init__(self, class_augmentations):
        super(DefaultTargetTransform, self).__init__()
        self.class_augmentations = class_augmentations

        self._augmentations = dict((augmentation, i + 1)
            for (i, augmentation) in enumerate(class_augmentations))
        self._augmentations[None] = 0

    def __call__(self, target):
        assert isinstance(target, tuple) and len(target) == 2
        label, augmentation = target
        return (label, self._augmentations[augmentation])

class Categorical(TargetTransform):
    def __init__(self, num_classes=None):
        super(Categorical, self).__init__()
        self.num_classes = num_classes
        self._classes = None
        self._labels = None

    def reset(self):
        self._classes = None
        self._labels = None

    @property
    def classes(self):
        if self._classes is None:
            self._classes = defaultdict(None)
            if self.num_classes is None:
                default_factory = lambda: len(self._classes)
            else:
                default_factory = lambda: self.labels[len(self._classes)]
            self._classes.default_factory = default_factory
        if (self.num_classes is not None) and (len(self._classes) > self.num_classes):
            raise ValueError('The number of individual labels ({0}) is greater '
                'than the number of classes defined by `num_classes` '
                '({1}).'.format(len(self._classes), self.num_classes))
        return self._classes

    @property
    def labels(self):
        if (self._labels is None) and (self.num_classes is not None):
            # TODO: Replace torch.randperm with seed-friendly counterpart
            self._labels = torch.randperm(self.num_classes).tolist()
        return self._labels

    def __call__(self, target):
        return self.classes[target]

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self.num_classes or '')


class FixedCategory(object):
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, index):
        return (index, self.transform)

    def __repr__(self):
        return ('{0}({1})'.format(self.__class__.__name__, self.transform))


class ClassDataset(object):
    def __init__(self, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, class_augmentations=None):
        if meta_train + meta_val + meta_test == 0:
            if meta_split is None:
                raise ValueError('The meta-split is undefined. Use either the '
                    'argument `meta_train=True` (or `meta_val`/`meta_test`), or '
                    'the argument `meta_split="train"` (or "val"/"test").')
            elif meta_split not in ['train', 'val', 'test']:
                raise ValueError('Unknown meta-split name `{0}`. The meta-split '
                    'must be in [`train`, `val`, `test`].'.format(meta_split))
            meta_train = (meta_split == 'train')
            meta_val = (meta_split == 'val')
            meta_test = (meta_split == 'test')
        elif meta_train + meta_val + meta_test > 1:
            raise ValueError('Multiple arguments among `meta_train`, `meta_val` '
                'and `meta_test` are set to `True`. Exactly one must be set to '
                '`True`.')
        self.meta_train = meta_train
        self.meta_val = meta_val
        self.meta_test = meta_test
        self._meta_split = meta_split

        if class_augmentations is not None:
            if not isinstance(class_augmentations, list):
                raise TypeError('Unknown type for `class_augmentations`. '
                    'Expected `list`, got `{0}`.'.format(type(class_augmentations)))
            unique_augmentations = OrderedSet()
            for augmentations in class_augmentations:
                for transform in augmentations:
                    if transform in unique_augmentations:
                        warnings.warn('The class augmentation `{0}` already '
                            'exists in the list of class augmentations (`{1}`). '
                            'To avoid any duplicate, this transformation is '
                            'ignored.'.format(transform, repr(transform)),
                            UserWarning, stacklevel=2)
                    unique_augmentations.add(transform)
            class_augmentations = list(unique_augmentations)
        else:
            class_augmentations = []
        self.class_augmentations = class_augmentations

    def get_class_augmentation(self, index):
        transform_index = (index // self.num_classes) - 1
        if transform_index < 0:
            return None
        return self.class_augmentations[transform_index]

    def get_transform(self, index, transform=None):
        class_transform = self.get_class_augmentation(index)
        if class_transform is None:
            return transform
        if transform is None:
            return class_transform
        return Compose([class_transform, transform])

    def get_target_transform(self, index):
        class_transform = self.get_class_augmentation(index)
        return FixedCategory(class_transform)

    @property
    def meta_split(self):
        if self._meta_split is None:
            if self.meta_train:
                self._meta_split = 'train'
            elif self.meta_val:
                self._meta_split = 'val'
            elif self.meta_test:
                self._meta_split = 'test'
            else:
                raise NotImplementedError()
        return self._meta_split

    def __getitem__(self, index):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_classes * (len(self.class_augmentations) + 1)


class MetaDataset(object):
    def __init__(self, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, target_transform=None, dataset_transform=None):
        if meta_train + meta_val + meta_test == 0:
            if meta_split is None:
                raise ValueError('The meta-split is undefined. Use either the '
                    'argument `meta_train=True` (or `meta_val`/`meta_test`), or '
                    'the argument `meta_split="train"` (or "val"/"test").')
            elif meta_split not in ['train', 'val', 'test']:
                raise ValueError('Unknown meta-split name `{0}`. The meta-split '
                    'must be in [`train`, `val`, `test`].'.format(meta_split))
            meta_train = (meta_split == 'train')
            meta_val = (meta_split == 'val')
            meta_test = (meta_split == 'test')
        elif meta_train + meta_val + meta_test > 1:
            raise ValueError('Multiple arguments among `meta_train`, `meta_val` '
                'and `meta_test` are set to `True`. Exactly one must be set to '
                '`True`.')
        self.meta_train = meta_train
        self.meta_val = meta_val
        self.meta_test = meta_test
        self._meta_split = meta_split
        self.target_transform = target_transform
        self.dataset_transform = dataset_transform
        self.seed()

    @property
    def meta_split(self):
        if self._meta_split is None:
            if self.meta_train:
                self._meta_split = 'train'
            elif self.meta_val:
                self._meta_split = 'val'
            elif self.meta_test:
                self._meta_split = 'test'
            else:
                raise NotImplementedError()
        return self._meta_split

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed=seed)
        # Seed the dataset transform
        _seed_dataset_transform(self.dataset_transform, seed=seed)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def sample_task(self):
        index = self.np_random.randint(len(self))
        return self[index]

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class CombinationMetaDataset(MetaDataset):
    def __init__(self, dataset, num_classes_per_task, target_transform=None,
                 dataset_transform=None):
        if not isinstance(num_classes_per_task, int):
            raise TypeError('Unknown type for `num_classes_per_task`. Expected '
                '`int`, got `{0}`.'.format(type(num_classes_per_task)))
        self.dataset = dataset
        self.num_classes_per_task = num_classes_per_task
        # If no target_transform, then use a default target transform that
        # is well behaved for the `default_collate` function (assign class
        # augmentations ot integers).
        if target_transform is None:
            target_transform = DefaultTargetTransform(dataset.class_augmentations)

        super(CombinationMetaDataset, self).__init__(meta_train=dataset.meta_train,
            meta_val=dataset.meta_val, meta_test=dataset.meta_test,
            meta_split=dataset.meta_split, target_transform=target_transform,
            dataset_transform=dataset_transform)

    def __iter__(self):
        num_classes = len(self.dataset)
        for index in combinations(num_classes, self.num_classes_per_task):
            yield self[index]

    def sample_task(self):
        index = self.np_random.choice(len(self.dataset),
            size=self.num_classes_per_task, replace=False)
        return self[tuple(index)]

    def __getitem__(self, index):
        if isinstance(index, int):
            raise ValueError('The index of a `CombinationMetaDataset` must be '
                'a tuple of integers, and not an integer. For example, call '
                '`dataset[({0})]` to get a task with classes from 0 to {1} '
                '(got `{2}`).'.format(', '.join([str(idx)
                for idx in range(self.num_classes_per_task)]),
                self.num_classes_per_task - 1, index))
        assert len(index) == self.num_classes_per_task
        datasets = [self.dataset[i] for i in index]
        # Use deepcopy on `Categorical` target transforms, to avoid any side
        # effect across tasks.
        task = ConcatTask(datasets, self.num_classes_per_task,
            target_transform=wrap_transform(self.target_transform,
            self._copy_categorical, transform_type=Categorical))

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task

    def _copy_categorical(self, transform):
        assert isinstance(transform, Categorical)
        transform.reset()
        if transform.num_classes is None:
            transform.num_classes = self.num_classes_per_task
        return deepcopy(transform)

    def __len__(self):
        num_classes, length = len(self.dataset), 1
        for i in range(1, self.num_classes_per_task + 1):
            length *= (num_classes - i + 1) / i

        if length > sys.maxsize:
            warnings.warn('The number of possible tasks in {0} is '
                'combinatorially large (equal to C({1}, {2})), and exceeds '
                'machine precision. Setting the length of the dataset to the '
                'maximum integer value, which undervalues the actual number of '
                'possible tasks in the dataset. Therefore the value returned by '
                '`len(dataset)` should not be trusted as being representative '
                'of the true number of tasks.'.format(self, len(self.dataset),
                self.num_classes_per_task), UserWarning, stacklevel=2)
            length = sys.maxsize
        return int(length)


def _seed_dataset_transform(transform, seed=None):
    if isinstance(transform, Compose):
        for subtransform in transform.transforms:
            _seed_dataset_transform(subtransform, seed=seed)
    elif hasattr(transform, 'seed'):
        transform.seed(seed=seed)

class Omniglot(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 use_vinyals_split=True, transform=None, target_transform=None,
                 dataset_transform=None, class_augmentations=None, download=False):
        dataset = OmniglotClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test,
            use_vinyals_split=use_vinyals_split, transform=transform,
            meta_split=meta_split, class_augmentations=class_augmentations,
            download=download)
        super(Omniglot, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class OmniglotClassDataset(ClassDataset):
    folder = 'omniglot'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    filename = 'data.hdf5'
    filename_labels = '{0}{1}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, use_vinyals_split=True, transform=None,
                 class_augmentations=None, download=False):
        super(OmniglotClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        if self.meta_val and (not use_vinyals_split):
            raise ValueError('Trying to use the meta-validation without the '
                'Vinyals split. You must set `use_vinyals_split=True` to use '
                'the meta-validation split.')

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.use_vinyals_split = use_vinyals_split
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename)
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format('vinyals_' if use_vinyals_split else '',
            self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Omniglot integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        character_name = '/'.join(self.labels[index % self.num_classes])
        data = self.data[character_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return OmniglotDataset(index, data, character_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.split_filename, 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):
        if self._check_integrity():
            return

        for name in self.zips_md5:
            zip_filename = '{0}.zip'.format(name)
            filename = os.path.join(self.root, zip_filename)
            if os.path.isfile(filename):
                continue

            url = '{0}/{1}'.format(self.download_url_prefix, zip_filename)
            download_url(url, self.root, zip_filename, self.zips_md5[name])

            with zipfile.ZipFile(filename, 'r') as f:
                f.extractall(self.root)

        filename = os.path.join(self.root, self.filename)
        with h5py.File(filename, 'w') as f:
            for name in self.zips_md5:
                group = f.create_group(name)

                alphabets = list_dir(os.path.join(self.root, name))
                characters = [(name, alphabet, character) for alphabet in alphabets
                    for character in list_dir(os.path.join(self.root, name, alphabet))]

                split = 'train' if name == 'images_background' else 'test'
                labels_filename = os.path.join(self.root,
                    self.filename_labels.format('', split))
                with open(labels_filename, 'w') as f_labels:
                    labels = sorted(characters)
                    json.dump(labels, f_labels)

                for _, alphabet, character in characters:
                    filenames = glob.glob(os.path.join(self.root, name,
                        alphabet, character, '*.png'))
                    dataset = group.create_dataset('{0}/{1}'.format(alphabet,
                        character), (len(filenames), 105, 105), dtype='uint8')

                    for i, char_filename in enumerate(filenames):
                        image = Image.open(char_filename, mode='r').convert('L')
                        dataset[i] = ImageOps.invert(image)

                shutil.rmtree(os.path.join(self.root, name))

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename_labels.format(
                'vinyals_', split))
            data = get_asset(self.folder, '{0}.json'.format(split), dtype='json')

            with open(filename, 'w') as f:
                labels = sorted([('images_{0}'.format(name), alphabet, character)
                    for (name, alphabets) in data.items()
                    for (alphabet, characters) in alphabets.items()
                    for character in characters])
                json.dump(labels, f)

class OmniglotDataset(Dataset):
    def __init__(self, index, data, character_name,
                 transform=None, target_transform=None):
        super(OmniglotDataset, self).__init__(index, transform=transform,
                                              target_transform=target_transform)
        self.data = data
        self.character_name = character_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.character_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)

class MiniImagenet(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = MiniImagenetClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(MiniImagenet, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class MiniImagenetClassDataset(ClassDataset):
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    folder = 'miniimagenet'
    gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    gz_filename = 'mini-imagenet.tar.gz'
    gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    pkl_filename = 'mini-imagenet-cache-{0}.pkl'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(MiniImagenetClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('MiniImagenet integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return MiniImagenetDataset(index, data, class_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        if self._check_integrity():
            return

        gdd.download_file_from_google_drive(self.gdrive_id, self.root,
            self.gz_filename, md5=self.gz_md5)

        filename = os.path.join(self.root, self.gz_filename)
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            pkl_filename = os.path.join(self.root, self.pkl_filename.format(split))
            if not os.path.isfile(pkl_filename):
                raise IOError()
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f)
                images, classes = data['image_data'], data['class_dict']

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)

            if os.path.isfile(pkl_filename):
                os.remove(pkl_filename)


#Dataloaders to load the datasets above
class BatchMetaCollate(object):

    def __init__(self, collate_fn):
        super().__init__()
        self.collate_fn = collate_fn

    def collate_task(self, task):
        if isinstance(task, TorchDataset):
            return self.collate_fn([task[idx] for idx in range(len(task))])
        elif isinstance(task, OrderedDict):
            return OrderedDict([(key, self.collate_task(subtask))
                for (key, subtask) in task.items()])
        else:
            raise NotImplementedError()

    def __call__(self, batch):
        return self.collate_fn([self.collate_task(task) for task in batch])


class MiniImagenetDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(MiniImagenetDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)



class MetaDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        if collate_fn is None:
            collate_fn = no_collate

        if isinstance(dataset, CombinationMetaDataset) and (sampler is None):
            if shuffle:
                sampler = CombinationRandomSampler(dataset)
            else:
                sampler = CombinationSequentialSampler(dataset)
            shuffle = False

        super(MetaDataLoader, self).__init__(dataset, batch_size=batch_size,
            shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn)


class BatchMetaDataLoader(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        collate_fn = BatchMetaCollate(default_collate)

        super(BatchMetaDataLoader, self).__init__(dataset,
            batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=None, num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)


class CombinationSequentialSampler(SequentialSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        super(CombinationSequentialSampler, self).__init__(data_source)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        return combinations(range(num_classes), num_classes_per_task)


class CombinationRandomSampler(RandomSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        # Temporarily disable the warning if the length of the length of the 
        # dataset exceeds the machine precision. This avoids getting this
        # warning shown with MetaDataLoader, even though MetaDataLoader itself
        # does not use the length of the dataset.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super(CombinationRandomSampler, self).__init__(data_source,
                                                           replacement=True)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        for _ in combinations(range(num_classes), num_classes_per_task):
            yield tuple(random.sample(range(num_classes), num_classes_per_task))

class Rotation(object):
    def __init__(self, angle, resample=False, expand=False, center=None):
        super(Rotation, self).__init__()
        if isinstance(angle, (list, tuple)):
            self._angles = angle
            self.angle = None
        else:
            self._angles = [angle]
            self.angle = angle
            if angle % 360 == 0:
                import warnings
                warnings.warn('Applying a rotation of {0} degrees (`{1}`) as a '
                    'class augmentation on a dataset is equivalent to the original '
                    'dataset.'.format(angle, self), UserWarning, stacklevel=2)

        self.resample = resample
        self.expand = expand
        self.center = center

    def __iter__(self):
        return iter(Rotation(angle, resample=self.resample, expand=self.expand,
            center=self.center) for angle in self._angles)

    def __call__(self, image):
        if self.angle is None:
            raise ValueError('The value of the angle is unspecified.')
        # QKFIX: Explicitly compute the pixel fill value due to an
        # incompatibility between Torchvision 0.5 and Pillow 7.0.0
        # https://github.com/pytorch/vision/issues/1759#issuecomment-583826810
        # Will be fixed in Torchvision 0.6
        fill = tuple([0] * len(image.getbands()))
        return F.rotate(image, self.angle % 360, F.InterpolationMode.NEAREST,
                        self.expand, self.center, fill=fill)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if (self.angle is None) or (other.angle is None):
            return self._angles == other._angles
        return (self.angle % 360) == (other.angle % 360)

    def __repr__(self):
        if self.angle is None:
            return 'Rotation({0})'.format(', '.join(map(str, self._angles)))
        else:
            return 'Rotation({0})'.format(self.angle % 360)

    def __str__(self):
        if self.angle is None:
            return 'Rotation({0})'.format(', '.join(map(str, self._angles)))
        else:
            return 'Rotation({0})'.format(self.angle)

def helper_with_default(klass, folder, shots, ways, shuffle=True,
                        test_shots=None, seed=None, defaults={}, **kwargs):
    if 'num_classes_per_task' in kwargs:
        warnings.warn('Both arguments `ways` and `num_classes_per_task` were '
            'set in the helper function for the number of classes per task. '
            'Ignoring the argument `ways`.', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = defaults.get('transform', ToTensor())
    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = defaults.get('target_transform',
                                                  Categorical(ways))
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = defaults.get('class_augmentations', None)
    if test_shots is None:
        test_shots = shots
    dataset = klass(folder, num_classes_per_task=ways, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=test_shots)
    dataset.seed(seed)

    return dataset

def omniglot(folder, shots, ways, shuffle=True, test_shots=None,
             seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(28), ToTensor()]),
        'class_augmentations': [Rotation([90, 180, 270])]
    }

    return helper_with_default(Omniglot, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)