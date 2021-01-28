import torch
import logging
import os
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import ptdarts.utils as utils
from data_loader.weighted_data_loader import loadCIFARData, getWeightedDataLoaders
from loss import calculate_weighted_loss
from ptdarts.models import augment_cnn as augment_cnn
from utils import progress
import ptdarts.genotypes as gt

'''this trains the new set of weights (W2) by minimizing the weighted training loss (given a set of weights per training sample)'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initial_model(config):
    genotype_str = "Genotype(normal=[[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], [('skip_connect', 0), ('dil_conv_3x3', 2)], " \
                   "[('sep_conv_3x3', 1), ('skip_connect', 0)], [('sep_conv_3x3', 1), ('skip_connect', 0)]]," \
                   "normal_concat=range(2, 6)," \
                   "reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)]," \
                   "[('skip_connect', 3), ('max_pool_3x3', 0)], [('skip_connect', 2), ('max_pool_3x3', 0)]]," \
                   "reduce_concat=range(2, 6))"
    genotype = gt.from_str(genotype_str)
    use_aux = config['aux_weight'] > 0
    model = augment_cnn.AugmentCNN(config['input_size'], config['input_channels'], config['init_channels'], config['n_classes'],
                                   config['layers'],
                                   use_aux, genotype)
    return model


def main(train_queue, config):
    cudnn.benchmark = True
    cudnn.enabled = True

    criterion = nn.CrossEntropyLoss(reduction='none')
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
        train(train_queue, model, criterion, optimizer, config['grad_clip'])
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        torch.save(model.state_dict(), os.path.join(script_dir + config['save'], 'network_weights_2.pt'))

def train(train_queue, model, criterion, optimizer, grad_clip):
  objs = utils.AverageMeter()
  model.train()

  for step, (input, target, weights) in enumerate(train_queue):
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