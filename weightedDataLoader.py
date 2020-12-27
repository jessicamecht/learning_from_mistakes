import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

class WeightedCIFAR(CIFAR10):
    '''initializes the weights fos all images with 0.5 for further evaluation'''
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        weight = 0.5
        return img, target, weight

def loadCIFARData(root = 'data'):
    '''loads the cifar dataset and creates train, test and validation splits'''
    train_data = WeightedCIFAR(root=root, train=True, download=True, transform=transform_train)
    test_data = WeightedCIFAR(root=root, train=False, download=True, transform=transform_test)
    torch.manual_seed(43)
    val_data_size = 512
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
