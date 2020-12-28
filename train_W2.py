import torch
from DARTS_CNN.train import infer
import logging
import sys, os
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import DARTS_CNN.utils as utils
import DARTS_CNN.genotypes as genotypes
from DARTS_CNN.model import NetworkCIFAR as Network
from weighted_data_loader import loadCIFARData, getWeightedDataLoaders

#TODO switch to argparser later in the process
CIFAR_CLASSES = 10
layers = 8
auxiliary = False
init_channels = 16
arch = 'DARTS'
learning_rate = 0.025
momentum = 0.9
weight_decay = 3e-4
epochs = 600
drop_path_prob = 0.2
save = 'EXP'
grad_clip = 5
report_freq = 50


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    cudnn.benchmark = True
    cudnn.enabled = True

    genotype = eval("genotypes.%s" % arch)
    model = Network(init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)
    model = model.to(device)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    train_data, valid_data, test_data = loadCIFARData()
    train_queue, valid_queue, test_loader = getWeightedDataLoaders(train_data, valid_data, test_data)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    for epoch in range(epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = drop_path_prob * epoch / epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        #valid_acc, valid_obj = infer(valid_queue, model, criterion)
        #logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(save, 'weights.pt'))

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target, weights) in enumerate(train_queue):
    input = input.to(device)
    target = target.to(device)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    #calculate the weighted loss
    loss = torch.mean(criterion(logits, target) * weights)
    print(loss.shape)
    if auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == "__main__":
    main()