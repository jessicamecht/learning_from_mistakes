import torch
import numpy as np
import torch.nn as nn
import os, sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DARTS_CNN.test import get_initial_model
from coefficient_update.model import LinearRegression
from loss import calculate_weighted_loss
from visual_similarity.visual_similarity import visual_validation_similarity
from label_similarity.label_similarity import measure_label_similarity
from weight_samples.sample_weights import calculate_similarities
from utils import create_visual_feature_extractor_model


from weighted_data_loader import loadCIFARData, getWeightedDataLoaders

from weight_samples.sample_weights import calculate_similarities
'''TODO:
1. Use existing weighting scheme 
2. minimize the validation loss of the weighted predictions s^Tw * x == y '''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(train_queue, val_queue):
    inputDim = 1
    outputDim = 1
    learningRate = 0.01
    epochs = 100

    model = LinearRegression(inputDim, outputDim)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    feature_extractor_model = create_visual_feature_extractor_model(init=True)

    train(model, criterion, optimizer, train_queue, val_queue, epochs, feature_extractor_model)


def train(model, criterion, optimizer, train_queue, val_queue, epochs, feature_extractor_model):

    pred_model = get_initial_model(16, 10, 20, True)

    for epoch in epochs:
        for step, val_input, val_target in enumerate(val_queue):
            val_input, val_target = val_input.to(device), val_target.to(device)

            for step, (train_input, train_target, _) in enumerate(train_queue):

                ##get similarities
                # for each training example batch, calculate the similarity to the validation samples and
                # combine them to the overall training instance weight
                label_similarity = measure_label_similarity(train_target, val_target).to(device)

                visual_similarity = visual_validation_similarity(val_input, train_input, feature_extractor_model).to(device)
                logits, _ = pred_model(train_input)
                loss = criterion(logits, train_target)
                loss = loss.to(device)

                similarities = calculate_similarities(loss, visual_similarity, label_similarity)
                logits = model(similarities)

                input = train_input.to(device)
                target = train_target.to(device)

                weights = predict_weights(logits, train_input)

                optimizer.zero_grad()

                # calculate the weighted loss
                loss, logits = calculate_weighted_loss(input, target, model, criterion, weights)

                val_logits = model(val_input)
                preds = criterion(val_logits, val_target)

                val_loss, val_logits = calculate_weighted_loss(val_input, val_target, model, criterion, weights)

                loss.backward()

                optimizer.step()

                print('epoch {}, loss {}'.format(epoch, loss.item()))

def predict_weights(similarities, data):
    return np.matmul(similarities, data)

if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)

    main(train_queue, val_queue)

