import os
from ptdarts import search
from tensorboardX import SummaryWriter
import torch
from data_loader.weighted_data_loader import loadCIFARData, getWeightedDataLoaders, create_clean_initial_weights


if __name__ == "__main__":
    writer = SummaryWriter(log_dir="tensorboard")

    train_data, val_data, test_data = loadCIFARData()
    train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data, batch_size=26)

    print("Start Architecture Search")
    path = os.path.join('searchs')
    _ = search.main(train_queue, val_queue, path, writer=writer)