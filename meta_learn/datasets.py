import random
import warnings
from collections import OrderedDict
from itertools import combinations

from meta_learn.data_utils import (
    Categorical,
    ClassSplitter,
    CombinationMetaDataset,
    MiniImagenet,
    Omniglot,
    Sinusoid,
)

from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, ToTensor

# TODO: Reduce all of this to bare minimum for understandability reasons
# which may require a bit of re-implementation

#############################################
# Helper Functions
#############################################
def get_sinusoid(shot):
    train = Sinusoid(num_samples_per_task=10000)
    train = ClassSplitter(
        train, shuffle=True, num_train_per_class=shot, num_test_per_class=shot
    )

    test = Sinusoid(num_samples_per_task=100000)
    test = ClassSplitter(
        test, shuffle=False, num_train_per_class=shot, num_test_per_class=shot
    )
    return train, test


def get_omniglot(data_path, way, download):
    transform = Compose([Resize(28), ToTensor()])
    target_transforms = Categorical(way)
    train = Omniglot(
        data_path,
        num_classes_per_task=way,
        transform=transform,
        target_transform=target_transforms,
        meta_train=True,
        download=download,
    )
    test = Omniglot(
        data_path,
        num_classes_per_task=way,
        transform=transform,
        target_transform=target_transforms,
        meta_val=True,
        download=download,
    )

    train = ClassSplitter(
        train, shuffle=True, num_train_per_class=way, num_test_per_class=way
    )

    test = ClassSplitter(
        test, shuffle=True, num_train_per_class=way, num_test_per_class=way
    )
    return train, test


def get_mini_imaget(data_path, way, download):
    transform = Compose([Resize(28), ToTensor()])
    target_transforms = Categorical(way)
    train = MiniImagenet(
        data_path,
        num_classes_per_task=way,
        transform=transform,
        target_transform=target_transforms,
        meta_train=True,
        download=download,
    )
    test = MiniImagenet(
        data_path,
        num_classes_per_task=way,
        transform=transform,
        target_transform=target_transforms,
        meta_val=True,
        download=download,
    )

    train = ClassSplitter(
        train, shuffle=True, num_train_per_class=way, num_test_per_class=way
    )

    test = ClassSplitter(
        test, shuffle=True, num_train_per_class=way, num_test_per_class=way
    )
    return train, test


#########################################
# Dataloaders - for meta data
#########################################
class BatchMetaCollate(object):
    def __init__(self, collate_fn):
        super().__init__()
        self.collate_fn = collate_fn

    def collate_task(self, task):
        if isinstance(task, Dataset):
            return self.collate_fn([task[idx] for idx in range(len(task))])
        elif isinstance(task, OrderedDict):
            return OrderedDict(
                [(key, self.collate_task(subtask)) for (key, subtask) in task.items()]
            )
        else:
            raise NotImplementedError()

    def __call__(self, batch):
        return self.collate_fn([self.collate_task(task) for task in batch])


class BatchMetaDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=True,
        sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):

        # Set up sampler and collate function for meta datasets
        if isinstance(dataset, CombinationMetaDataset) and (sampler is None):
            if shuffle:
                sampler = CombinationRandomSampler(dataset)
            else:
                sampler = CombinationSequentialSampler(dataset)
            shuffle = False

        collate_fn = BatchMetaCollate(default_collate)

        # Call Dataloader initialization function
        super(BatchMetaDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


#########################################
# Random sample utilities
#########################################


class CombinationSequentialSampler(SequentialSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError(
                "Expected `data_source` to be an instance of "
                "`CombinationMetaDataset`, but found "
                "{0}".format(type(data_source))
            )
        super(CombinationSequentialSampler, self).__init__(data_source)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        return combinations(range(num_classes), num_classes_per_task)


class CombinationRandomSampler(RandomSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError(
                "Expected `data_source` to be an instance of "
                "`CombinationMetaDataset`, but found "
                "{0}".format(type(data_source))
            )
        # Temporarily disable the warning if the length of the length of the
        # dataset exceeds the machine precision. This avoids getting this
        # warning shown with MetaDataLoader, even though MetaDataLoader itself
        # does not use the length of the dataset.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(CombinationRandomSampler, self).__init__(
                data_source, replacement=True
            )

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        for _ in combinations(range(num_classes), num_classes_per_task):
            yield tuple(random.sample(range(num_classes), num_classes_per_task))
