import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_weighted_loss(logits, target, criterion, weights, reduce=True):
    '''calculates the weighted loss for each input example, and then averages everything in the end
    :param input torch tensor of size (number of examples, feature_dimension_1, ...)
    :param target torch tensor of size (number of examples,)
    :param criterion torch nn loss function
    :param weights torch tensor of size (number of examples,)
    :returns tuple of averaged loss and class probabilities as torch tensors of size (number of examples,)'''
    loss = criterion(logits, target)
    weights = weights.cpu()
    weights = torch.tensor(np.array(weights).astype(float)).to(device)
    weighted_loss_individual = loss.float() * weights.float()
    loss = torch.mean(weighted_loss_individual) if reduce else weighted_loss_individual
    return loss