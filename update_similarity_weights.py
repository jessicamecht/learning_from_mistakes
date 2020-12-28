import torch
import torch.nn as nn
from DARTS_CNN import test
from visual_similarity.visual_similarity import visual_validation_similarity
from label_similarity.label_similarity import measure_label_similarity
from sample_weights import sample_weights
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer_similarities(train_queue, val_queue):
  '''calls calculations for predictive performance, label and visual similarity and calculates the overall similarity score'''

  model = test.get_initial_model()
  model = model.to(device)

  criterion = nn.CrossEntropyLoss(reduction='none')
  criterion = criterion.to(device)

  model.eval()

  for step, data_label in enumerate(val_queue):
    val_input, val_target = data_label[0].to(device), data_label[1].to(device)
    with torch.no_grad(): # we do not need gradients for these calculations
      input = val_input.to(device)
      target = val_target.to(device)

      # the predictive performance is the negative cross entropy loss
      # TODO check if this is actually correct
      logits, _ = model(input)
      loss = criterion(logits, target)

      for i, elem in enumerate(train_queue):
        print("Validation Batch: ", step, " for training batch: ", i)
        train_input, train_target = elem[0].to(device), elem[1].to(device)

        # for each training example batch, calculate the similarity to the validation samples and
        # combine them to the overall training instance weight
        label_similarity = measure_label_similarity(train_target, val_target)
        visual_similarity = visual_validation_similarity(val_input, train_input)
        weights = sample_weights(loss, visual_similarity, label_similarity)

        #get the indices in the dataset for which the weights were calculated in this iteration
        indices = np.array(train_data.indices)[list(range(i*train_target.shape[0],(i+1)*train_target.shape[0]))]
        weights = weights.cpu()

        #update the weights in the dataset
        train_data.dataset.regenerate_instance_weights(indices, weights)


if __name__ =="__main__":
    weights = infer_similarities()
