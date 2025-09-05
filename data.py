# data.py

"""
typing      -> Tuple: This is a type hint for a tuple.
torch       -> DataLoader: This is a class from the PyTorch library for loading data.
torchvision -> datasets: This is a module from the PyTorch library for working with datasets.
torchvision -> transforms: This is a module from the PyTorch library for transforming data.
"""

from typing import Tuple
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets


# MNIST Loader:
# - Transform each raw image into a Pytorch tensor with pred. shape
# - Then, wrap the dataset in DataLoader for mini-batches
# - Return the train/test loaders for ease of training loop

def get_mnist_loaders(batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    # After this first line, each image is [1, 28, 28]
    
    transform = T.ToTensor()
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # Turn shuffle off for testing for reproducibility, but randomization is needed for SGD
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # x.shape = [64, 1, 28, 28], x.dtype == torch.float32
    # y.shape = [64], y.dtype == torch.int64
    return train_loader, test_loader

def get_fashion_loaders(batch_size: int = 64, num_workers: int = 4):
    transform = T.ToTensor()
    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_cifar10_loaders(batch_size: int = 64, num_workers: int = 4):
    transform = T.ToTensor()  # keep simple; you can add normalization later
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_loaders(dataset: str, batch_size: int = 64, num_workers: int = 4):
    dataset = dataset.lower()
    if dataset == "mnist":
        return get_mnist_loaders(batch_size=batch_size, num_workers=num_workers)
    if dataset in ("fashion", "fashionmnist", "fashion-mnist"):
        return get_fashion_loaders(batch_size=batch_size, num_workers=num_workers)
    if dataset in ("cifar10", "cifar"):
        return get_cifar10_loaders(batch_size=batch_size, num_workers=num_workers)
    raise ValueError(f"Unknown dataset: {dataset}")

def infer_in_dim(dataset: str) -> int:
    dataset = dataset.lower()
    if dataset in ("mnist", "fashion", "fashionmnist", "fashion-mnist"):
        return 28 * 28 * 1
    if dataset in ("cifar10", "cifar"):
        return 32 * 32 * 3
    raise ValueError(f"Unknown dataset for in_dim: {dataset}")

def get_class_names(dataset: str) -> list[str]:
    dataset = dataset.lower()
    if dataset == "mnist":
        return datasets.MNIST(root='./data', train=False, download=True).classes
    if dataset in ("fashion", "fashionmnist", "fashion-mnist"):
        return datasets.FashionMNIST(root='./data', train=False, download=True).classes
    if dataset in ("cifar10", "cifar"):
        return datasets.CIFAR10(root='./data', train=False, download=True).classes
    raise ValueError(f"Unknown dataset for class names: {dataset}")
