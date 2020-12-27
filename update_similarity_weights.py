import torch
from torch.autograd import Variable
import torch.nn as nn
from DARTS_CNN import test, utils
from weightedDataLoader import loadCIFARData, getWeightedDataLoaders
from visual_similarity.visual_similarity import visual_validation_similarity
from label_similarity.label_similarity import measure_label_similarity
from sample_weights import sample_weights


def update_architecture_weights():
    '''this method calculates the label, visual and predictive similarities for a given validation set'''
    model = test.get_initial_model()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion#.cuda()

    train_data, val_data, test_data = loadCIFARData()
    train_loader, val_loader, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
    print('got weighted data loaders')
    pred_performance, visual_similarity, label_similarity = infer_similarities(val_loader, model, criterion, train_loader)
    weights = sample_weights(pred_performance, visual_similarity, label_similarity)
    print(weights.shape, 'weights')

def infer_similarities(val_queue, model, criterion, train_queue):
  losses = torch.empty(1,1)
  model.eval()

  elem = next(iter(val_queue))
  input, target = elem[0], elem[1]

  visual_sim = torch.empty(input.shape[0], input.shape[0], 2048, 1, 1)
  label_sim = torch.empty(target.shape[0], target.shape[0])

  for step, data_label in enumerate(val_queue):
    if step > 1:
        break
    val_input, val_target = data_label[0], data_label[1]
    with torch.no_grad():
      print("Validation Batch: ", step)
      val_input = val_input.permute(0, 3, 1, 2).float() / 255.
      input = Variable(val_input)#.cuda()
      target = Variable(val_target)#.cuda(async=True)

      logits, _ = model(input)
      loss = criterion(logits, target)
      loss_tensor =  torch.tensor([[loss.data.item()]])
      print(loss_tensor.shape)
      losses = torch.cat((losses,loss_tensor))

      for i, elem in enumerate(train_queue):
        if i > 5:
            break
        train_input, train_target = elem[0], elem[1]
        train_input = train_input.permute(0, 3, 1, 2).float() / 255.
        print("Calculate label similarity")
        label_similarity = measure_label_similarity(train_target, val_target)
        print("Calculate visual similarity")
        visual_similarity = visual_validation_similarity(val_input, train_input)
        visual_sim = torch.cat((visual_sim, visual_similarity))
        label_sim = torch.cat((label_sim,label_similarity))

  return objs, visual_sim, label_sim


if __name__ =="__main__":
    update_architecture_weights()