import torch
import logging
import os
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import ptdarts.utils as utils
from data_loader.weighted_data_loader import loadCIFARData, getWeightedDataLoaders
from loss import calculate_weighted_loss
from DARTS_CNN import test
from utils import progress

'''this trains the new set of weights (W2) by minimizing the weighted training loss (given a set of weights per training sample)'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(train_queue, config):
    cudnn.benchmark = True
    cudnn.enabled = True

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)
    model = test.get_initial_model(36, 10, 20, True)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    epochs = config['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    for epoch in range(epochs):
        model.drop_path_prob = config['drop_path_prob'] * epoch / epochs
        train(train_queue, model, criterion, optimizer, config['grad_clip'])
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        torch.save(model.state_dict(), os.path.join(script_dir + config['save'], 'network_weights_2.pt'))

def train(train_queue, model, criterion, optimizer, grad_clip):
  objs = utils.AverageMeter()
  model.train()
  for step, (input, target, weights) in enumerate(train_queue):
    if step > 0:
        break
    print('W2 calculation', progress(step, len(train_queue), train_queue))
    input = input.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    logits, _ = model(input)
    loss = calculate_weighted_loss(logits, target, criterion, weights)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    n = input.size(0)
    objs.update(loss.data.item(), n)
  return objs.avg



if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
    main(train_queue)