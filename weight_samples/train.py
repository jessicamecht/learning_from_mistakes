import torch
import logging
import os
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import ptdarts.utils as utils
from data_loader.weighted_data_loader import loadCIFARData, getWeightedDataLoaders
from loss import calculate_weighted_loss
from utils import progress, initial_model

'''this trains the new set of weights by minimizing the (weighted) training loss'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(train_queue, val_queue,  config, weight_samples=False, save_name='network_weights'):
    cudnn.benchmark = True
    cudnn.enabled = True

    criterion = nn.CrossEntropyLoss(reduction='none') if weight_samples else nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    model = initial_model(config)
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
        loss, top1, top5 = train(train_queue, model, criterion, optimizer, config['grad_clip'], weight_samples)
        print("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, epochs, top1))

        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        torch.save(model.state_dict(), os.path.join(script_dir + config['save'], save_name + '.pt'))

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.eval()
    for val_data, val_target in val_queue:
        N = val_data.size(0)
        logits, loss = model(val_data)
        prec1, prec5 = utils.accuracy(logits, val_target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
    print("Validation: Loss {losses.avg:.3f} "
          "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(losses=losses,
                                                               top1=top1, top5=top5))

def train(train_queue, model, criterion, optimizer, grad_clip, weight_samples=False):
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  losses = utils.AverageMeter()
  model.train()

  for step, (input, target, weights) in enumerate(train_queue):
    print('Weights calculation', progress(step, len(train_queue), train_queue))
    input = input.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    logits, _ = model(input)
    loss = calculate_weighted_loss(logits, target, criterion, weights) if weight_samples else criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    N = input.size(0)
    losses.update(loss.item(), N)
    top1.update(top1.item(), N)
    top5.update(top5.item(), N)

  return losses.avg, top1.avg, top5.avg



if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
    main(train_queue)