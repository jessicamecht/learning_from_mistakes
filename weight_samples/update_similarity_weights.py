import torch
import torch.nn as nn
import numpy as np
from weight_samples.visual_similarity.visual_similarity import visual_validation_similarity
from weight_samples.label_similarity.label_similarity import measure_label_similarity
from weight_samples.sample_weights import sample_weights
from utils import progress
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_similarity_weights(train_data, train_queue, val_queue, config):
  '''calls calculations for predictive performance, label and visual similarity and calculates the overall similarity score and saves it to file for the dataset
  :param train_data
  :param train_queue training loader
  :param val_queue validation loader'''

  ######## Setup DARTS pretrained model
  model = initial_model(config)
  model = model.to(device)

  criterion = nn.CrossEntropyLoss(reduction='none')
  criterion = criterion.to(device)

  model.eval()
  val_data = next(iter(val_queue))
  val_input, val_target = val_data[0].to(device), val_data[1].to(device)
  with torch.no_grad(): # we do not need gradients for these calculations

    ######## Predictive Performance
    # the predictive performance is the negative cross entropy loss for all validation examples
    logits, _ = model(val_input)
    loss = criterion(logits, val_target) * -1 # TODO check if this is actually correct
    loss = loss.to(device)
    assert(loss.shape == val_target.shape)

    for i, elem in enumerate(train_queue):
      train_input, train_target = elem[0].to(device), elem[1].to(device)
      print('Similarity weight calculation', progress(i, len(train_queue), train_queue))

      ######### Label Similarity
      # for each training example batch, calculate the similarity to the validation samples and
      # combine them to the overall training instance weight
      label_similarity = measure_label_similarity(val_target, train_target).to(device)
      ######### Visual Similarity
      visual_similarity = visual_validation_similarity(val_input, train_input, init=True).to(device)

      r = torch.ones(val_target.shape[0], 1).to(device)
      weights = sample_weights(loss, visual_similarity, label_similarity, r)
      weights = weights.reshape(weights.shape[0],)

      #get the indices in the dataset for which the weights were calculated in this iteration
      indices = np.array(train_data.indices)[list(range(i*train_target.shape[0],(i+1)*train_target.shape[0]))]

      weights = weights.cpu()

      #update the weights in the dataset
      assert(indices.shape == weights.shape)
      train_data.dataset.regenerate_instance_weights(indices, weights)



if __name__ =="__main__":
    weights = infer_similarities()
