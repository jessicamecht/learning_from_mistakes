import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from typing import Any, Callable, Optional, Tuple
from csv import reader
import os
import numpy as np

class WeightedCIFAR(CIFAR10):
    '''initializes the weights for all images with 0.5 for further evaluation'''

    def __init__(self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            instance_weights = None) -> None:
        super(WeightedCIFAR, self).__init__(root, train, transform, target_transform, download)
        file_path = os.path.join(self.root, self.base_folder, "instance_weights.csv")
        self.instance_weights = []
        if train:
            with open(file_path, 'r') as f:
                csv_reader = reader(f)
                for row in csv_reader:
                    self.instance_weights.extend(row)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if len(self.instance_weights) > 0:
            weight = self.instance_weights[index]
        return img, target, weight

    def regenerate_instance_weights(self, update_idxs, update_values):
        instance_weight_np = np.array(self.instance_weights)
        instance_weight_np[update_idxs] = update_values
        self.instance_weights = instance_weight_np

def loadCIFARData(root = 'data'):
    '''loads the cifar dataset and creates train, test and validation splits'''
    train_data = WeightedCIFAR(root=root, train=True, download=True, transform=transform_train)
    test_data = WeightedCIFAR(root=root, train=False, download=True, transform=transform_test)
    torch.manual_seed(43)
    val_data_size = 25000
    train_size = len(train_data) - val_data_size
    train_data, val_data = torch.utils.data.dataset.random_split(train_data, [train_size, val_data_size])
    return train_data, val_data, test_data

def getWeightedDataLoaders(train_data, val_data, test_data,batch_size = 10):
    '''creates dataloader for train, test and validation sets including a weight variable'''
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_loader, val_loader, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
    print(next(iter(test_data)))
