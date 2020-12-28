import torch
import update_similarity_weights
import train_W2
from weighted_data_loader import loadCIFARData, getWeightedDataLoaders
import gc


def main():
    gc.collect()
    torch.cuda.empty_cache()
    # load data
    train_data, val_data, test_data = loadCIFARData()
    train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)

    # First Stage: uses pretrained DARTS Architecture (weights W1), calculates similarity weights
    # for each training sample and updates them in the dataset instance_weights.csv
    update_similarity_weights.infer_similarities(train_data, train_queue, val_queue)

    # Second Stage: based on the calculated weights for each training instance, calculates a second
    # set of weights given the DARTS architecture by minimizing weighted training loss
    train_W2.main(train_queue)

    # Third Stage.1: based on the new set of weights, update the architecture A by minimizing the validation loss

    # Third Stage.2: update image embedding V by minimizing the validation loss

    # Third Stage.3: update coefficient vector r by minimizing the validation loss

if __name__ == "__main__":
    main()