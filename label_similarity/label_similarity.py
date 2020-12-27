import torch
import os, sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weightedDataLoader import loadCIFARData, getWeightedDataLoaders

def measure_label_similarity(training_example_targets, validation_example_targets):
    label_similarity = torch.empty(training_example_targets.shape[0], validation_example_targets.shape[0], 1)
    for i, train_target in enumerate(training_example_targets):
        for j, val_target in enumerate(validation_example_targets):
            label_similarity[i,j] = train_target == val_target
    return label_similarity


if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_loader, val_loader, test_loader = getWeightedDataLoaders(train_data, val_data, test_data, batch_size=10)
    train_imgs, train_targets, train_weights = next(iter(train_loader))
    val_imgs, val_targets, val_weights = next(iter(val_loader))
    val_imgs = val_imgs.permute(0, 3, 1, 2).float() / 255.
    train_imgs = train_imgs.permute(0, 3, 1, 2).float() / 255.

    print(measure_label_similarity(train_targets, val_targets).shape)


