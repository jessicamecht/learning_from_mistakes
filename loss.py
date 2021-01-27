import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_weighted_loss(input, target, model, criterion, weights):
    # for similarities s s^Tw * x == y
    #Todo this is wrong
    logits = model(input)
    loss = criterion(logits, target)
    weights = weights.cpu()
    weights = torch.tensor(np.array(weights).astype(float)).to(device)
    weighted_loss_individual = loss.float() * weights.float()
    loss = torch.mean(weighted_loss_individual)
    return loss, logits

def calculate_weighted_loss_logistic(input, x, target, model, criterion, weights, actual_model):
    # for similarities s s^Tw * x == y

    #TODO check if I have to weight the prediction or the logits?
    logits_linreg = model(input) * x
    #TODO predict with actual model to compare
    preds = criterion(logits, target)
    weights = weights.cpu()
    weights = torch.tensor(np.array(weights).astype(float)).to(device)
    weighted_loss_individual = preds.float() * weights.float()
    loss = torch.mean(weighted_loss_individual)
    return loss, logits