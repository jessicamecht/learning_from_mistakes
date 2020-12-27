import torch
from torch.autograd import Variable
import torch.nn as nn
from DARTS_CNN import test, utils
from weightedDataLoader import loadCIFARData, getWeightedDataLoaders
from visual_similarity.visual_similarity import visual_validation_similarity
from label_similarity.label_similarity import measure_label_similarity
from sample_weights import sample_weights


def run_initial_architecture_weights():
    model = test.get_initial_model()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion#.cuda()

    train_data, val_data, test_data = loadCIFARData()
    train_loader, val_loader, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
    pred_performance, visual_similarity, label_similarity = infer_similarities(val_loader, model, criterion, train_loader)
    weights = sample_weights(pred_performance, visual_similarity, label_similarity)
    print(weights.shape)

def infer_similarities(val_queue, model, criterion, train_queue):
  objs = utils.AvgrageMeter()
  vis_sim = []
  label_sim = []
  model.eval()

  for step, data_label in enumerate(val_queue):
    val_input, val_target = data_label[0], data_label[1]
    with torch.no_grad():
      val_input = val_input.permute(0, 3, 1, 2).float() / 255.
      input = Variable(val_input)#.cuda()
      target = Variable(val_target)#.cuda(async=True)

      logits, _ = model(input)
      loss = criterion(logits, target)
      visual_sim = []
      label_sim = []

      for elem in train_queue:
        train_input, train_target = elem[0], elem[1]
        train_input = train_input.permute(0, 3, 1, 2).float() / 255.
        visual_similarity = visual_validation_similarity(val_input, train_input)
        label_similarity = measure_label_similarity(train_target, val_target)
        visual_sim.append(visual_similarity)
        label_sim.append(label_similarity)

      n = input.size(0)
      objs.update(loss.data.item(), n)
      vis_sim.extend(visual_sim)
      label_sim.extend(label_sim)

  return objs, vis_sim, label_sim


if __name__ =="__main__":
    run_initial_architecture_weights()