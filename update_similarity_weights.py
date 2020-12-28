import torch
import torch.nn as nn
import sys
import logging
from DARTS_CNN import test
from weighted_data_loader import loadCIFARData, getWeightedDataLoaders
from visual_similarity.visual_similarity import visual_validation_similarity
from label_similarity.label_similarity import measure_label_similarity
from sample_weights import sample_weights
import numpy as np
from torch.autograd import Variable


def infer_similarities():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if device != 'cpu':
    torch.cuda.set_device(device)

  model = test.get_initial_model()
  model = model.to(device)
  criterion = nn.CrossEntropyLoss(reduction='none')
  criterion = criterion.to(device)
  train_data, val_data, test_data = loadCIFARData()
  train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
  model.eval()

  for step, data_label in enumerate(val_queue):
    val_input, val_target = data_label[0].to(device), data_label[1].to(device)
    with torch.no_grad():
      input = val_input.to(device)
      target = val_target.to(device)

      #predictive performance
      logits, _ = model(input)
      loss = criterion(logits, target)

      for i, elem in enumerate(train_queue):
        print("Validation Batch: ", step, " for training batch: ", i)
        train_input, train_target = Variable(elem[0]).to(device), Variable(elem[1]).to(device)
        label_similarity = measure_label_similarity(train_target, val_target)
        visual_similarity = visual_validation_similarity(val_input, train_input)
        weights = sample_weights(loss, visual_similarity, label_similarity)
        indices = np.array(train_data.indices)
        indices = indices[list(range(i*train_target.shape[0],(i+1)*train_target.shape[0]))]
        weights = weights.cpu()
        train_data.dataset.regenerate_instance_weights(indices, weights)


if __name__ =="__main__":
    weights = infer_similarities()
