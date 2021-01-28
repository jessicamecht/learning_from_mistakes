import os, sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weighted_data_loader import loadCIFARData, getWeightedDataLoaders

def measure_label_similarity(training_example_targets, validation_example_targets):
    '''checks for each training and test label if they are the same'''
    res = training_example_targets.unsqueeze(1) - validation_example_targets
    res = (res == 0).float()
    assert(res.shape[0] == training_example_targets.shape[0] and res.shape[1] == validation_example_targets.shape[0])
    return res


if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_loader, val_loader, test_loader = getWeightedDataLoaders(train_data, val_data, test_data, batch_size=10)
    train_imgs, train_targets, train_weights = next(iter(train_loader))
    val_imgs, val_targets, val_weights = next(iter(val_loader))
    print(measure_label_similarity(train_targets, val_targets).shape)


