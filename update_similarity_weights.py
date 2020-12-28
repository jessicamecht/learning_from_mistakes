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

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  torch.cuda.set_device(0)

  #TODO this is highly stripped down for testing purposes on CPU
  model = test.get_initial_model()
  model = model.cuda()
  criterion = nn.CrossEntropyLoss(reduction='none').to('cpu')
  criterion = criterion.cuda()
  train_data, val_data, test_data = loadCIFARData()
  train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
  model.eval()

  for step, data_label in enumerate(val_queue):
    val_input, val_target = Variable(data_label[0]).cuda(), Variable(data_label[1]).cuda()
    with torch.no_grad():
      print("Validation Batch: ", step)
      val_input = val_input.permute(0, 3, 1, 2).float() / 255.
      input = Variable(val_input).cuda()
      target = Variable(val_target).cuda()

      #predictive performance
      logits, _ = model(input)
      loss = criterion(logits, target)

      for i, elem in enumerate(train_queue):
        train_input, train_target = Variable(elem[0]).cuda(), Variable(elem[1]).cuda()
        train_input = train_input.permute(0, 3, 1, 2).float() / 255.
        print("Calculate label similarity")
        label_similarity = measure_label_similarity(train_target, val_target)
        print("Calculate visual similarity")
        visual_similarity = visual_validation_similarity(val_input, train_input)
        weights = sample_weights(loss, visual_similarity, label_similarity)
        indices = np.array(train_data.indices)
        indices = indices[list(range(i,train_target.shape[0]+1))]
        weights = weights.cpu()
        train_data.dataset.regenerate_instance_weights(indices, weights)
  #TODO update weights in CSV


if __name__ =="__main__":
    weights = infer_similarities()
