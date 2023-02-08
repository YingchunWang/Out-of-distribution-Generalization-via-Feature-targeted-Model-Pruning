from typing import List, Union

import os
import copy
import numpy as np
from PIL import Image, ImageColor, ImageOps
from scipy.stats import bernoulli
import torch
import h5py
import torchvision
import torchvision.transforms as transforms
# from lacuna import Lacuna10, Lacuna100, Small_Lacuna10, Small_Binary_Lacuna10, Small_Lacuna5
# from Small_CIFAR10 import Small_CIFAR10, Small_Binary_CIFAR10, Small_CIFAR5
# from Small_MNIST import Small_MNIST, Small_Binary_MNIST
# from TinyImageNet import TinyImageNet_pretrain, TinyImageNet_finetune, TinyImageNet_finetune5
# from IPython import embed
from matplotlib.colors import to_rgb
class Object_data(torch.utils.data.Dataset):

    def __init__(self, data_set,  p, val ,transform):
        super(Object_data, self).__init__()
        self.transform = transform
        CLASSES = ['person', 'giraffe', 'truck', 'cup', 'airplane', 'horse', 'elephant', 'umbrella', 'zebra', 'train']
        self.labels = []
        self.images = None
        for one in range(len(CLASSES)):
            tep = np.load('./{}/unbiased_{}_file_{}.pkl.npy'.format(data_set,val,CLASSES[one]))
            if isinstance(self.images, np.ndarray):
                self.images = np.concatenate((self.images, tep), axis = 0)
            else:
                self.images = tep
            self.labels += len(tep)*[one]

        self.labels = np.array(self.labels)
        self.domain_labels = np.ones(self.labels.shape)
        if p:
            images2 = None
            labels2 = []
            for one in range(len(CLASSES)):
                tep = np.load('./{}/biased_{}_file_{}.pkl.npy'.format(data_set,val, CLASSES[one]))
                if isinstance(images2, np.ndarray):
                    images2 = np.concatenate((images2, tep), axis=0)
                else:
                    images2 = tep
                labels2+= len(tep)*[one]

            labels2 = np.array(labels2)
            domain_labels2 = np.zeros(labels2.shape)
            indexes = np.random.choice(len(self.labels), int(p*len(self.labels)), replace = False)
            self.images[indexes] = images2[indexes]
            self.labels[indexes] = labels2[indexes]
            self.domain_labels[indexes] = domain_labels2[indexes]

        self.labels = self.labels.astype(np.int)
        # self.images = self.images.astype(np.float)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label, domain_label = self.images[idx], self.labels[idx], self.domain_labels[idx]
        # img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        # label = torch.from_numpy(label)
        # all_label = torch.tensor([label, domain_label])
        return img, label, domain_label





color_dict = {
  0: to_rgb('red'),
  1: to_rgb('green'),
  2: to_rgb('#FFFF00'),
  3: to_rgb('#802A2A'),
  4: to_rgb('#A020F0'),
  5: to_rgb('#0000FF'),
  6: to_rgb('#708069'),
  7: to_rgb('#FF6100'),
  8: to_rgb('#00C78C'),
  9: to_rgb('#B03060')
}

def manual_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_DATASETS = {}


def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn


def _get_mnist_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test


def _get_lacuna_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test


def _get_cifar_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=4, fill=(125, 123, 113)),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_mnist_transforms(augment=True):
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    return apply_transform, apply_transform



def _get_imagenet_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test


def _get_mix_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test


@_add_dataset
def cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment = augment)
    train_set = MNIST_colored(root=root, train=True, download=True, transform=transform_train)
    test_set = MNIST_colored(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def mnist_colored(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment = augment)
    train_set = MNIST_colored(root=root, train=True, download=True, transform=transform_train)
    test_set = MNIST_colored(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def fmnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = torchvision.datasets.FashionMNIST(root=root, train= True, download=True, transform=transform_train)
    test_set = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform_train)
    return train_set, test_set

class MNIST_colored(torchvision.datasets.MNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform= None,
            target_transform= None,
            download: bool = False,
    ) -> None:
        super(MNIST_colored, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# @_add_dataset
# def small_cifar5(root, augment=False):
#     transform_train, transform_test = _get_cifar_transforms(augment=augment)
#     train_set = Small_CIFAR5(root=root, train=True, transform=transform_train)
#     test_set = Small_CIFAR5(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def small_cifar10(root, augment=False):
#     transform_train, transform_test = _get_cifar_transforms(augment=augment)
#     train_set = Small_CIFAR10(root=root, train=True, transform=transform_train)
#     test_set = Small_CIFAR10(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def small_binary_cifar10(root, augment=False):
#     transform_train, transform_test = _get_cifar_transforms(augment=augment)
#     train_set = Small_Binary_CIFAR10(root=root, train=True, transform=transform_train)
#     test_set = Small_Binary_CIFAR10(root=root, train=False, transform=transform_test)
#     return train_set, test_set

#
# @_add_dataset
# def small_mnist(root, augment=False):
#     transform_train, transform_test = _get_mnist_transforms(augment=augment)
#     train_set = Small_MNIST(root=root, train=True, transform=transform_train)
#     test_set = Small_MNIST(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def small_binary_mnist(root, augment=False):
#     transform_train, transform_test = _get_mnist_transforms(augment=augment)
#     train_set = Small_Binary_MNIST(root=root, train=True, transform=transform_train)
#     test_set = Small_Binary_MNIST(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def lacuna100(root, augment=False):
#     transform_train, transform_test = _get_lacuna_transforms(augment=augment)
#     train_set = Lacuna100(root=root, train=True, transform=transform_train)
#     test_set = Lacuna100(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def lacuna10(root, augment=False):
#     transform_train, transform_test = _get_lacuna_transforms(augment=augment)
#     train_set = Lacuna10(root=root, train=True, transform=transform_train)
#     test_set = Lacuna10(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def small_lacuna5(root, augment=False):
#     transform_train, transform_test = _get_lacuna_transforms(augment=augment)
#     train_set = Small_Lacuna5(root=root, train=True, transform=transform_train)
#     test_set = Small_Lacuna5(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def small_lacuna10(root, augment=False):
#     transform_train, transform_test = _get_lacuna_transforms(augment=augment)
#     train_set = Small_Lacuna10(root=root, train=True, transform=transform_train)
#     test_set = Small_Lacuna10(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def small_binary_lacuna10(root, augment=False):
#     transform_train, transform_test = _get_lacuna_transforms(augment=augment)
#     train_set = Small_Binary_Lacuna10(root=root, train=True, transform=transform_train)
#     test_set = Small_Binary_Lacuna10(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def tinyimagenet_pretrain(root, augment=False):
#     transform_train, transform_test = _get_imagenet_transforms(augment=augment)
#     train_set = TinyImageNet_pretrain(root=root, train=True, transform=transform_train)
#     test_set = TinyImageNet_pretrain(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def tinyimagenet_finetune(root, augment=False):
#     transform_train, transform_test = _get_imagenet_transforms(augment=augment)
#     train_set = TinyImageNet_finetune(root=root, train=True, transform=transform_train)
#     test_set = TinyImageNet_finetune(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def tinyimagenet_finetune5(root, augment=False):
#     transform_train, transform_test = _get_imagenet_transforms(augment=augment)
#     train_set = TinyImageNet_finetune5(root=root, train=True, transform=transform_train)
#     test_set = TinyImageNet_finetune5(root=root, train=False, transform=transform_test)
#     return train_set, test_set
#
#
# @_add_dataset
# def mix10(root, aug
# ment=False):
#     transform_train, transform_test = _get_mix_transforms(augment=augment)
#     lacuna_train_set = Lacuna10(root=root, train=True, transform=transform_train)
#     lacuna_test_set = Lacuna10(root=root, train=False, transform=transform_test)
#     cifar_train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform_train)
#     cifar_test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform_test)
#
#     lacuna_train_set.targets = np.array(lacuna_train_set.targets)
#     lacuna_test_set.targets = np.array(lacuna_test_set.targets)
#     cifar_train_set.targets = np.array(cifar_train_set.targets)
#     cifar_test_set.targets = np.array(cifar_test_set.targets)
#
#     lacuna_train_set.data = lacuna_train_set.data[:, ::2, ::2, :]
#     lacuna_test_set.data = lacuna_test_set.data[:, ::2, ::2, :]
#
#     classes = np.arange(5)
#     for c in classes:
#         lacuna_train_class_len = np.sum(lacuna_train_set.targets == c)
#         lacuna_train_set.data[lacuna_train_set.targets == c] = cifar_train_set.data[cifar_train_set.targets == c] \
#             [:lacuna_train_class_len, :, :, :]
#         lacuna_test_class_len = np.sum(lacuna_test_set.targets == c)
#         lacuna_test_set.data[lacuna_test_set.targets == c] = cifar_test_set.data[cifar_test_set.targets == c] \
#             [:lacuna_test_class_len, :, :, :]
#     return lacuna_train_set, lacuna_test_set
#
#
# @_add_dataset
# def mix100(root, augment=False):
#     transform_train, transform_test = _get_mix_transforms(augment=augment)
#     lacuna_train_set = Lacuna100(root=root, train=True, transform=transform_train)
#     lacuna_test_set = Lacuna100(root=root, train=False, transform=transform_test)
#     cifar_train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=transform_train)
#     cifar_test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=transform_test)
#
#     lacuna_train_set.targets = np.array(lacuna_train_set.targets)
#     lacuna_test_set.targets = np.array(lacuna_test_set.targets)
#     cifar_train_set.targets = np.array(cifar_train_set.targets)
#     cifar_test_set.targets = np.array(cifar_test_set.targets)
#
#     lacuna_train_set.data = lacuna_train_set.data[:, ::2, ::2, :]
#     lacuna_test_set.data = lacuna_test_set.data[:, ::2, ::2, :]
#
#     classes = np.arange(50)
#     for c in classes:
#         lacuna_train_class_len = np.sum(lacuna_train_set.targets == c)
#         lacuna_train_set.data[lacuna_train_set.targets == c] = cifar_train_set.data[cifar_train_set.targets == c] \
#             [:lacuna_train_class_len, :, :, :]
#         lacuna_test_class_len = np.sum(lacuna_test_set.targets == c)
#         lacuna_test_set.data[lacuna_test_set.targets == c] = cifar_test_set.data[cifar_test_set.targets == c] \
#             [:lacuna_test_class_len, :, :, :]
#     return lacuna_train_set, lacuna_test_set


def replace_indexes(dataset: torch.utils.data.Dataset, indexes: Union[List[int], np.ndarray], seed=0,
                    only_mark: bool = False):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(list(set(range(len(dataset))) - set(indexes)), size=len(indexes))
        dataset.data[indexes] = dataset.data[new_indexes]
        dataset.targets[indexes] = dataset.targets[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        dataset.targets[indexes] = - dataset.targets[indexes] - 1


def replace_class(dataset: torch.utils.data.Dataset, class_to_replace: int, num_indexes_to_replace: int = None,
                  seed: int = 0, only_mark: bool = False):
    indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)


def get_dis_loaders(dataset_name, class_to_replace: int = None, num_indexes_to_replace: int = None,
                indexes_to_replace: List[int] = None, seed: int = 1, only_mark: bool = False, root: str = './datasets',
                batch_size=128, shuffle=True,
                **dataset_kwargs):
    '''

    :param dataset_name: Name of dataset to use
    :param class_to_replace: If not None, specifies which class to replace completely or partially
    :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
                                   `num_indexes_to_replace` samples
    :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
                               indexes_to_replace can be specidied.
    :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
                 always in the same order
    :param root: Root directory to initialize the dataset
    :param batch_size: Batch size of data loader
    :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
    :param dataset_kwargs: Extra arguments to pass to the dataset init.
    :return: The train_loader and test_loader
    '''
    manual_seed(seed)
    if root is None:
        root = os.path.expanduser('~/data')
    train_set, test_set = _DATASETS[dataset_name](root, **dataset_kwargs)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    valid_idx = np.where(train_set.targets == class_to_replace)[0]
    other_idx = list(set(range(len(train_set))) - set(valid_idx))
    other_idx = np.random.choice(other_idx, len(valid_idx), replace=False)
    need_idx = list(valid_idx)+list(other_idx)

    train_set_copy = copy.deepcopy(train_set)
    train_set.data = train_set_copy.data[need_idx]
    train_set.targets = train_set_copy.targets[need_idx]
    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return train_loader




def get_dis_ratated_loaders(dataset_name, degree_to_replace: int = None, num_indexes_to_replace: int = None,
                indexes_to_replace: List[int] = None, seed: int = 1, only_mark: bool = False, root: str = './datasets',
                batch_size=128, shuffle=True,
                **dataset_kwargs):
    '''

    :param dataset_name: Name of dataset to use
    :param class_to_replace: If not None, specifies which class to replace completely or partially
    :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
                                   `num_indexes_to_replace` samples
    :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
                               indexes_to_replace can be specidied.
    :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
                 always in the same order
    :param root: Root directory to initialize the dataset
    :param batch_size: Batch size of data loader
    :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
    :param dataset_kwargs: Extra arguments to pass to the dataset init.
    :return: The train_loader and test_loader
    '''
    manual_seed(seed)
    if root is None:
        root = os.path.expanduser('~/data')
    train_set, test_set = _DATASETS[dataset_name](root, **dataset_kwargs)
    train_set.targets = torch.tensor(train_set.targets)
    test_set.targets = torch.tensor(test_set.targets)

    for i in range(3):
        if (i+1)*90 == degree_to_replace:
            pass
        else:
            data_new = torch.load('{}_data_all_rotated_{}.pt'.format(dataset_name,90*(i+1)))
            if isinstance(data_new.data, type(torch.tensor([]))):
                train_set.data = torch.cat((train_set.data, data_new.data), 0)
            else:
                train_set.data = np.vstack((train_set.data, data_new.data))
            train_set.targets = torch.cat((train_set.targets, data_new.targets), 0)
            data_new = torch.load('{}_test_data_all_rotated_{}.pt'.format(dataset_name,90*(i+1)))
            if isinstance(data_new.data, type(torch.tensor([]))):
                test_set.data = torch.cat((test_set.data, data_new.data), 0)
            else:
                test_set.data = np.vstack((test_set.data, data_new.data))
            test_set.targets = torch.cat((test_set.targets, data_new.targets), 0)

    data_rotated = torch.load('{}_data_all_rotated_{}.pt'.format(dataset_name,degree_to_replace))
    # data_rotated_original = copy.deepcopy(data_rotated)
    indexes = np.random.choice(len(train_set.targets), len(data_rotated.targets), replace=False)

    train_set.data = train_set.data[indexes]
    train_set.targets = train_set.targets[indexes]


    indexes_train = np.random.choice(len(indexes), int(len(indexes)*0.8), replace=False)
    indexes_test = list(set(range(len(indexes)))-set(indexes_train))

    data_rotated_train = copy.deepcopy(data_rotated)
    data_rotated_test = copy.deepcopy(data_rotated)


    # data_rotated.data = torch.cat((data_rotated.data, train_set.data[indexes]), 0)
    if isinstance(data_rotated.data, type(torch.tensor([]))):
        data_rotated_train.data = torch.cat((data_rotated.data[indexes_train], train_set.data[indexes_train]), 0)
        data_rotated_test.data = torch.cat((data_rotated.data[indexes_test], train_set.data[indexes_test]), 0)
    else:
        data_rotated_train.data = np.vstack((data_rotated.data[indexes_train], train_set.data[indexes_train]))
        data_rotated_test.data = np.vstack((data_rotated.data[indexes_test], train_set.data[indexes_test]))
    data_rotated_train.targets = torch.cat((torch.ones(len(indexes_train)), torch.zeros(len(indexes_train))), 0)
    data_rotated_test.targets = torch.cat((torch.ones(len(indexes_test)), torch.zeros(len(indexes_test))), 0)


    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    dis_loader = torch.utils.data.DataLoader(data_rotated_train, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    dis_test_loader = torch.utils.data.DataLoader(data_rotated_test, batch_size=batch_size, shuffle=shuffle,
                                             worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return dis_loader, dis_test_loader

def get_loaders(dataset_name, class_to_replace: int = None, num_indexes_to_replace: int = None,
                indexes_to_replace: List[int] = None, seed: int = 1, only_mark: bool = False, root: str = './datasets',
                batch_size=128, shuffle=True,
                **dataset_kwargs):
    '''

    :param dataset_name: Name of dataset to use
    :param class_to_replace: If not None, specifies which class to replace completely or partially
    :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
                                   `num_indexes_to_replace` samples
    :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
                               indexes_to_replace can be specidied.
    :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
                 always in the same order
    :param root: Root directory to initialize the dataset
    :param batch_size: Batch size of data loader
    :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
    :param dataset_kwargs: Extra arguments to pass to the dataset init.
    :return: The train_loader and test_loader
    '''
    manual_seed(seed)
    if root is None:
        root = os.path.expanduser('~/data')
    train_set, test_set = _DATASETS[dataset_name](root, **dataset_kwargs)
    test_set.targets = np.array(test_set.targets)

    valid_set = copy.deepcopy(train_set)
    rng = np.random.RandomState(seed)

    if class_to_replace:
        # remove_index = np.where(rest_set.targets == class_to_replace)[0]
        # rest_sampler = torch.utils.data.sampler.SubsetRandomSampler(remove_index)
        # unlearning_loader = torch.utils.data.DataLoader(rest_set, batch_size=batch_size, sampler = rest_sampler)

        valid_idx = np.where(train_set.targets == class_to_replace)[0]
    else:
        valid_idx = []
    # for i in range(max(train_set.targets) + 1):
    #     class_idx = np.where(train_set.targets == i)[0]
    #     valid_idx.append(rng.choice(class_idx, int(0 * len(class_idx)), replace=False))
    # valid_idx = np.hstack(valid_idx)
    train_idx = list(set(range(len(train_set))) - set(valid_idx))
    train_set_copy = copy.deepcopy(train_set)
    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError("Only one of `class_to_replace` and `indexes_to_replace` can be specified")
    if class_to_replace is not None:
        replace_class(train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed - 1, \
                      only_mark=only_mark)
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(dataset=train_set, indexes=indexes_to_replace, seed=seed - 1, only_mark=only_mark)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))



    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return train_loader, valid_loader, test_loader

def get_roated_loader(dataset_name, degree_to_replace: int = None, num_indexes_to_replace: int = None,
                seed: int = 1, only_mark: bool = False, root: str = './datasets',
                batch_size=128, shuffle=True,
                **dataset_kwargs):
    '''

    :param dataset_name: Name of dataset to use
    :param class_to_replace: If not None, specifies which class to replace completely or partially
    :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
                                   `num_indexes_to_replace` samples
    :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
                               indexes_to_replace can be specidied.
    :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
                 always in the same order
    :param root: Root directory to initialize the dataset
    :param batch_size: Batch size of data loader
    :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
    :param dataset_kwargs: Extra arguments to pass to the dataset init.
    :return: The train_loader and test_loader
    '''
    manual_seed(seed)
    if root is None:
        root = os.path.expanduser('~/data')
    train_set, test_set = _DATASETS[dataset_name](root, **dataset_kwargs)
    train_set.targets = torch.tensor(train_set.targets)
    test_set.targets = torch.tensor(test_set.targets)
    full_set = copy.deepcopy(train_set)
    # subset_size = 15000
    # indices = np.random.choice(len(train_set.targets), subset_size, replace=False)
    # for i in range(3):
    #     # indexes = np.random.choice(indices, 5000, replace=False)
    #     # indices = list(set(indices)-set(indexes))
    #     tep = copy.deepcopy(train_set)
    #     # tep.data = train_set.data[indexes]
    #     if isinstance(tep.data, type(torch.tensor([]))):
    #         tep.data = torch.rot90(tep.data, i+1, [1,2])
    #     else:
    #         tep.data = np.rot90(tep.data, i + 1, (1, 2))
    #     # tep.targets = train_set.targets[indexes]
    #     torch.save(tep, '{}_data_all_rotated_{}.pt'.format(dataset_name,90*(i+1)))
    # #
    # # subset_size = 3000
    # # indices = np.random.choice(len(test_set.targets), subset_size, replace=False)
    # for i in range(3):
    #     # indexes = np.random.choice(indices, 1000, replace=False)
    #     # indices = list(set(indices)-set(indexes))
    #     tep = copy.deepcopy(test_set)
    #     # tep.data = test_set.data[indexes]
    #     if isinstance(tep.data, type(torch.tensor([]))):
    #         tep.data = torch.rot90(tep.data, i + 1, [1, 2])
    #     else:
    #         tep.data = np.rot90(tep.data, i + 1, (1, 2))
    #     # tep.targets = test_set.targets[indexes]
    #     torch.save(tep, '{}_test_data_all_rotated_{}.pt'.format(dataset_name, 90*(i+1)))

    for i in range(3):
        if i == 0:
            pass
        else:
            data_new = torch.load('{}_data_all_rotated_{}.pt'.format(dataset_name,90*(i+1)))
            if isinstance(data_new.data, type(torch.tensor([]))):
                full_set.data = torch.cat((full_set.data, data_new.data),0)
            else:
                full_set.data = np.vstack((full_set.data, data_new.data))
            full_set.targets = torch.cat((full_set.targets, data_new.targets), 0)
            data_new = torch.load('{}_test_data_all_rotated_{}.pt'.format(dataset_name,90*(i+1)))
            if isinstance(data_new.data, type(torch.tensor([]))):
                test_set.data = torch.cat((test_set.data, data_new.data), 0)
            else:
                test_set.data = np.vstack((test_set.data, data_new.data))
            test_set.targets = torch.cat((test_set.targets, data_new.targets), 0)

    r_set_loader = []

    # valid_set = torch.load('{}_data_rotated_{}.pt'.format(dataset_name,degree_to_replace))
    # r_set_1 = torch.load('{}_data_rotated_{}.pt'.format(dataset_name, 180))
    # r_set_2 = torch.load('{}_data_rotated_{}.pt'.format(dataset_name, 270))

    # train_set.targets = np.array(train_set.targets)
    # test_set.targets = np.array(test_set.targets)
    rng = np.random.RandomState(seed)
    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    for i in range(3):
        data_tep = torch.load('{}_data_all'
                              '_rotated_{}.pt'.format(dataset_name,90*(i+1)))
        r_set_loader.append(torch.utils.data.DataLoader(data_tep, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
    #                                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    # r_set_loader1 = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
    #                                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    # r_set_loader2 = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
    #                                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    full_loader = torch.utils.data.DataLoader(full_set, batch_size=batch_size, shuffle=True,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return train_loader,  test_loader, r_set_loader, full_loader

def color_grayscale_arr(arr, label):
  assert arr.ndim == 2
  arr = arr.numpy()
  rgb = np.array(color_dict[int(label)])
  rgb = rgb * 255
  new_image = []

  for i in arr:
    row = []
    for j in i:
      tep = j/255
      tep = rgb + (np.array([255,255,255])-rgb)*tep
      row.append(tep)
    new_image.append(row)
  new_image = np.array(new_image).astype(np.uint8)
  file_name = '{}.png'.format(label)
  if os.path.exists(file_name):
    pass
  else:
    png = Image.fromarray(new_image)
    png.save(file_name)
    png_original = Image.fromarray(arr)
    png_original.save('original_'+file_name)
    print('save '+file_name)

  return new_image


def random_color_grayscale_arr(arr):
  assert arr.ndim == 2
  arr = arr.numpy()
  label = np.random.randint(0,10)
  rgb = np.array(color_dict[int(label)])
  rgb = rgb * 255
  new_image = []
  for i in arr:
    row = []
    for j in i:
      tep = j/255
      tep = rgb + (np.array([255,255,255])-rgb)*tep
      row.append(tep)
    new_image.append(row)
  new_image = np.array(new_image).astype(np.uint8)
  return new_image

def prepare_mixed_data(data_set_i, colored_set_i, p):
    data_set = copy.deepcopy(data_set_i)
    colored_set = copy.deepcopy(colored_set_i)
    data_num = len(data_set.targets)
    colored_train_num = int(data_num * p)
    colored_indexes = np.random.choice(data_num, colored_train_num, replace=False)
    rest_indexes = list(set(range(data_num)) - set(colored_indexes))
    data_set.data = data_set.data[rest_indexes]
    data_set.targets = data_set.targets[rest_indexes]
    colored_data = colored_set.data[colored_indexes]
    colored_label = colored_set.targets[colored_indexes]
    mixed_data = torch.cat([data_set.data, colored_data], dim=0)
    mixed_label = torch.cat([data_set.targets, colored_label], dim = 0)
    mixed_set = copy.deepcopy(data_set)
    mixed_set.data = mixed_data
    mixed_set.targets = mixed_label
    return mixed_set

def prepare_ood_dataset(dataset_name = 'mnist',p:float =0.8, seed: int =1, root: str = './datasets'):
    manual_seed(seed)
    if dataset_name == 'mnist':
        print('Preparing Colored MNIST')
        # train_set, test_set = _DATASETS[dataset_name](colored_mnist_dir)
        # train_colored = torch.load('Colored_Mnist_train.pt')
        # test_colored = torch.load('Colored_Mnist_test.pt')
        colored_train_set = torch.load('Colored_{}_train.pt'.format(dataset_name))
        colored_test_set = torch.load('Colored_{}_test.pt'.format(dataset_name))
        ood_train_set = torch.load('OOD_Colored_{}_train.pt'.format(dataset_name))
        ood_test_set = torch.load('OOD_Colored_{}_test.pt'.format(dataset_name))
        mixed_train = prepare_mixed_data(ood_train_set, colored_train_set, p)
        mixed_test = prepare_mixed_data(ood_test_set, colored_test_set, p)


    elif dataset_name == 'colored_object':

        print('Preparing Colored OBJECT')
        mixed_train = Object_data(file1_name='colored_object_unbiased_train.h5py',
                                  file2_name='colored_object_biased_train.h5py', p=p, transform=None)
        mixed_test = Object_data(file1_name='colored_object_unbiased_test.h5py',
                                 file2_name='colored_object_biased_test.h5py', p=p, transform=None)
        ood_test_set = Object_data(file1_name='colored_object_unbiased_test.h5py', file2_name=None, p=None,
                                   transform=None)
        colored_test_set = Object_data(file1_name='colored_object_biased_test.h5py', file2_name=None, p=None,
                                       transform=None)

    elif dataset_name == 'scene_object':

        print('Preparing scene OBJECT')
        mixed_train = Object_data(data_set=dataset_name, p=p, val='train',
                                  transform=None)
        mixed_test = Object_data(data_set=dataset_name, p=p, val='test',
                                  transform=None)
        ood_test_set = Object_data(data_set=dataset_name, p=None, val='test',
                                  transform=None)
        colored_test_set = Object_data(data_set=dataset_name, p=1, val='test',
                                  transform=None)
    # torch.save(mixed_train, 'Mixed_Mnist_train_{}.pt'.format(p))
    # torch.save(mixed_test, 'Mixed_Mnist_test_{}.pt'.format(p))
    return mixed_train, mixed_test, ood_test_set, colored_test_set

def full_colored_data(dataset_name = 'mnist', root: str = './datasets', ood: bool = False):
    colored_mnist_dir = os.path.join(root, dataset_name)
    train_set, test_set = _DATASETS[dataset_name](colored_mnist_dir)

    train_colored = []
    test_colored = []

    for one in range(len(train_set)):
        print("{}/{}".format(one, len(train_set)))
        if ood:
            train_colored.append(random_color_grayscale_arr(train_set.data[one]))
        else:
            train_colored.append(color_grayscale_arr(train_set.data[one], train_set.targets[one]))

    for one in range(len(test_set)):
        print("{}/{}".format(one, len(test_set)))
        if ood:
            test_colored.append(random_color_grayscale_arr(test_set.data[one]))
        else:
            test_colored.append(color_grayscale_arr(test_set.data[one], test_set.targets[one]))

    train_set.data = torch.tensor(np.array(train_colored))
    test_set.data = torch.tensor(np.array(test_colored))
    train_set.targets = train_set.targets.unsqueeze(1)
    test_set.targets = test_set.targets.unsqueeze(1)

    if ood:
        train_set.targets = torch.cat([train_set.targets, torch.ones(len(train_set.targets)).unsqueeze(1)], dim=1)
        test_set.targets = torch.cat([test_set.targets, torch.ones(len(test_set.targets)).unsqueeze(1)], dim=1)
        torch.save(train_set, 'OOD_Colored_{}_train.pt'.format(dataset_name))
        torch.save(test_set, 'OOD_Colored_{}_test.pt'.format(dataset_name))
    else:
        train_set.targets = torch.cat([train_set.targets, torch.zeros(len(train_set.targets)).unsqueeze(1)], dim=1)
        test_set.targets = torch.cat([test_set.targets, torch.zeros(len(test_set.targets)).unsqueeze(1)], dim=1)
        torch.save(train_set, 'Colored_{}_train.pt'.format(dataset_name))
        torch.save(test_set, 'Colored_{}_test.pt'.format(dataset_name))

# def prepare_colored_mnist(dataset_name = 'mnist',data_num:int =2000, seed: int =1, root: str = './datasets'):
#     manual_seed(seed)
#     colored_mnist_dir = os.path.join(root, 'ColoredMNIST')
#     print('Preparing Colored MNIST')
#     train_set, test_set = _DATASETS[dataset_name](colored_mnist_dir)
#     colored_mnist = train_set
#     original_mnist = copy.deepcopy(colored_mnist)
#     # indices = np.random.choice(len(colored_mnist.targets), data_num, replace=False)
#     # rest_indexes = list(set(range(len(colored_mnist))) - set(indices))
#     rest_indexes = np.random.choice(len(colored_mnist.targets), data_num, replace=False)
#     indices = list(set(range(len(colored_mnist))) - set(rest_indexes))
#     data_colored = []
#     for one in rest_indexes:
#         data_colored.append(color_grayscale_arr(colored_mnist.data[one], colored_mnist.targets[one]))
#     original_mnist.data = colored_mnist.data[indices].unsqueeze(dim=3)
#     original_mnist.data = original_mnist.data.expand(-1, -1, -1, 3)
#     original_mnist.targets = colored_mnist.targets[indices]
#     colored_mnist.data = torch.tensor(np.array(data_colored))
#     colored_mnist.targets = colored_mnist.targets[rest_indexes]
#     torch.save(colored_mnist, 'Colored_Mnist_{}.pt'.format(data_num))
#     torch.save(original_mnist, 'Colored_Mnist_{}_rest.pt'.format(data_num))
#
# def get_colored_mnist_loader(dataset_name, data_num: int = 5000,
#                 seed: int = 1, root: str = './datasets',
#                 batch_size=128, shuffle=True,
#                 **dataset_kwargs):
#     '''
#
#     :param dataset_name: Name of dataset to use
#     :param class_to_replace: If not None, specifies which class to replace completely or partially
#     :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
#                                    `num_indexes_to_replace` samples
#     :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
#                                indexes_to_replace can be specidied.
#     :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
#                  always in the same order
#     :param root: Root directory to initialize the dataset
#     :param batch_size: Batch size of data loader
#     :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
#     :param dataset_kwargs: Extra arguments to pass to the dataset init.
#     :return: The train_loader and test_loader
#     '''
#     # prepare_colored_mnist(dataset_name, data_num=data_num, seed=seed, root=root)
#     u_set = torch.load('Colored_Mnist_{}.pt'.format(data_num))
#     r_set = torch.load('Colored_Mnist_{}_rest.pt'.format(data_num))
#
#     full_set = copy.deepcopy(r_set)
#     full_set.data = torch.cat((full_set.data, u_set.data), 0)
#     full_set.targets = torch.cat((full_set.targets, u_set.targets), 0)
#
#     dis_set = copy.deepcopy(u_set)
#     indices = np.random.choice(len(r_set.targets), len(u_set), replace=False)
#     dis_set.data = torch.cat((dis_set.data, r_set.data[indices]), 0)
#     dis_set.targets = torch.cat((torch.zeros(len(u_set)), torch.ones(len(indices))), 0)
#
#     loader_args = {'num_workers': 0, 'pin_memory': False}
#
#     def _init_fn(worker_id):
#         np.random.seed(int(seed))
#
#     r_loader = torch.utils.data.DataLoader(r_set, batch_size=batch_size, shuffle=shuffle,
#                                                  worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#     # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
#     #                                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#     u_loader = torch.utils.data.DataLoader(u_set, batch_size=batch_size, shuffle=False,
#                                                   worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#
#     full_loader = torch.utils.data.DataLoader(full_set, batch_size=batch_size, shuffle=True,
#                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#     dis_loader = torch.utils.data.DataLoader(dis_set, batch_size=batch_size, shuffle=True,
#                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#
#     return r_loader, u_loader, full_loader, dis_loader
#
#
# def get_small_roated_loader(dataset_name, data_num: int = 5000,
#                 seed: int = 1, only_mark: bool = False, root: str = './datasets',
#                 batch_size=128, shuffle=True,
#                 **dataset_kwargs):
#     '''
#
#     :param dataset_name: Name of dataset to use
#     :param class_to_replace: If not None, specifies which class to replace completely or partially
#     :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
#                                    `num_indexes_to_replace` samples
#     :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
#                                indexes_to_replace can be specidied.
#     :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
#                  always in the same order
#     :param root: Root directory to initialize the dataset
#     :param batch_size: Batch size of data loader
#     :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
#     :param dataset_kwargs: Extra arguments to pass to the dataset init.
#     :return: The train_loader and test_loader
#     '''
#     manual_seed(seed)
#     if root is None:
#         root = os.path.expanduser('~/data')
#     train_set, test_set = _DATASETS[dataset_name](root, **dataset_kwargs)
#     train_set.targets = torch.tensor(train_set.targets)
#     test_set.targets = torch.tensor(test_set.targets)
#
#     fname = '{}_{}_data_small_rotated.pt'.format(dataset_name,data_num)
#     if not os.path.isfile(fname):
#         indices = np.random.choice(len(train_set.targets), data_num, replace=False)
#         tep = copy.deepcopy(train_set)
#         tep.data = train_set.data[indices]
#         if isinstance(tep.data, type(torch.tensor([]))):
#             tep.data = torch.rot90(tep.data, 1, [1,2])
#         else:
#             tep.data = np.rot90(tep.data, 1, (1, 2))
#         tep.targets = train_set.targets[indices]
#         torch.save(tep, '{}_{}_data_small_rotated.pt'.format(dataset_name,data_num))
#         rest_indexes = list(set(range(len(train_set)))-set(indices))
#         train_set.data = train_set.data[rest_indexes]
#         train_set.targets = train_set.targets[rest_indexes]
#         torch.save(train_set, '{}_{}_data_small_rest.pt'.format(dataset_name, data_num))
#
#     train_set = torch.load('{}_{}_data_small_rest.pt'.format(dataset_name, data_num))
#     data_new = torch.load('{}_{}_data_small_rotated.pt'.format(dataset_name, data_num))
#
#     full_set = copy.deepcopy(train_set)
#     if isinstance(full_set.data, type(torch.tensor([]))):
#         full_set.data = torch.cat((full_set.data, data_new.data), 0)
#     else:
#         full_set.data = np.vstack((full_set.data, data_new.data))
#     full_set.targets = torch.cat((full_set.targets, data_new.targets), 0)
#
#     dis_set = copy.deepcopy(data_new)
#     indices = np.random.choice(len(train_set.targets), len(data_new), replace=False)
#     if isinstance(dis_set.data, type(torch.tensor([]))):
#         dis_set.data = torch.cat((dis_set.data, train_set.data[indices]), 0)
#     else:
#         dis_set.data = np.vstack((dis_set.data, train_set.data[indices]))
#     dis_set.targets = torch.cat((torch.zeros(len(data_new)), torch.ones(len(indices))), 0)
#
#     rng = np.random.RandomState(seed)
#     loader_args = {'num_workers': 0, 'pin_memory': False}
#
#     def _init_fn(worker_id):
#         np.random.seed(int(seed))
#
#     unrotated_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
#                                                worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#     # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
#     #                                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#     rotated_loader = torch.utils.data.DataLoader(data_new, batch_size=batch_size, shuffle=False,
#                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#
#     full_loader = torch.utils.data.DataLoader(full_set, batch_size=batch_size, shuffle=True,
#                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#     dis_loader = torch.utils.data.DataLoader(dis_set, batch_size=batch_size, shuffle=True,
#                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
#
#
#
#
#     return unrotated_loader, rotated_loader, full_loader, dis_loader

if __name__ == '__main__':
    # tep = prepare_ood_colored_mnist('mnist', 0.8)
    # full_colored_data('mnist', ood=True)
    # full_colored_data('mnist', ood=False)
    # full_colored_data('mnist', ood=True)
    # full_colored_data('mnist', ood=False)
    # data_set_name = 'fmnist'
    # colored_train_set = torch.loaprepare_ood_datasetd('Colored_{}_train.pt'.format(data_set_name))
    # colored_test_set = torch.load('Colored_{}_test.pt'.format(data_set_name))
    # # train_set, test_set = _DATASETS[data_set_name]('./datasets')
    # #
    # ood_train_set = torch.load('OOD_Colored_{}_train.pt'.format(data_set_name))
    # ood_test_set = torch.load('OOD_Colored_{}_test.pt'.format(data_set_name))
    #
    # result = prepare_mixed_data(ood_train_set, colored_train_set, 0.8)
    result = prepare_ood_dataset('mnist')
    print('ok')