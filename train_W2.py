import torch
import logging
import numpy as np
import sys, os
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from DARTS_CNN import genotypes as genotypes
import DARTS_CNN.utils as utils
from DARTS_CNN.model import NetworkCIFAR as Network
from weighted_data_loader import loadCIFARData, getWeightedDataLoaders

'''this trains the new set of weights (W2) by minimizing the training loss given a set of weights per training sample'''

#TODO switch to argparser later in the process
CIFAR_CLASSES = 10
layers = 8
auxiliary = False
auxiliary_weight = 0.4
init_channels = 16
arch = 'DARTS'
learning_rate = 0.025
momentum = 0.9
weight_decay = 3e-4
epochs = 600
drop_path_prob = 0.2
save = './EXP'
grad_clip = 5
report_freq = 50


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(train_queue):
    cudnn.benchmark = True
    cudnn.enabled = True

    # load the model
    genotype = eval("genotypes.%s" % arch)
    model = Network(init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)
    model = model.to(device)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # set reduction to none to be able to weight them individually
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    for epoch in range(epochs):

        model.drop_path_prob = drop_path_prob * epoch / epochs

        train_obj = train(train_queue, model, criterion, optimizer)

        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])

        utils.save(model, os.path.join(save, 'weights_2.pt'))

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  model.train()

  for step, (input, target, weights) in enumerate(train_queue):
    input = input.to(device)
    target = target.to(device)

    optimizer.zero_grad()

    #calculate the weighted loss
    loss = calculate_weighted_loss(input, target, model, criterion, weights)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    n = input.size(0)
    objs.update(loss.data.item(), n)
    assert(not torch.isnan(loss).any())
    assert(not torch.isnan(input).any())
    assert(not torch.isnan(target).any())
    assert(not torch.isnan(weights).any())


    if step % report_freq == 0:
      logging.info('train %03d %e', step, objs.avg)

  return objs.avg

def calculate_weighted_loss(input, target, model, criterion, weights):
    logits = model(input)
    preds = criterion(logits, target)
    weights = torch.tensor(np.array(weights).astype(float)).to(device)
    weighted_loss_individual = preds.float() * weights.float()
    loss = torch.mean(weighted_loss_individual)
    return loss

if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
    main(train_queue)