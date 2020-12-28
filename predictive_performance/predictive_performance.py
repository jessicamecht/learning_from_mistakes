import torch
import torch.nn as nn
import os, sys
from torch.autograd import Variable
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_similarity import resnet_model
from weighted_data_loader import  loadCIFARData, getWeightedDataLoaders

#TODO remove this class since the predictive performance is the loss of the architecture


def measure_predictive_performance(predicted_validation_samples, validation_labels):
    loss = nn.CrossEntropyLoss()
    return -1 * loss(predicted_validation_samples, validation_labels)

def predict_label(images):
    '''loads a resnet pretrained model for CIFAR and predicts with the given set of weights'''
    resnet_50_model = resnet_model.resnet50(pretrained=True)
    resnet_50_model = resnet_50_model.cuda()
    #for p in resnet_50_model.parameters():
    #    p.requires_grad = False
    img_var = Variable(images).cuda()
    pred_var = resnet_50_model(img_var)
    pred = pred_var.data
    return pred

if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_loader, val_loader, test_loader = getWeightedDataLoaders(train_data, val_data, test_data, batch_size=10)
    train_imgs, train_targets, train_weights = next(iter(train_loader))
    val_imgs, val_targets, val_weights = next(iter(val_loader))
    val_imgs = val_imgs.permute(0, 3, 1, 2).float() / 255.
    train_imgs = train_imgs.permute(0, 3, 1, 2).float() / 255.

    predicted_validation_samples = predict_label(val_imgs)
    print(predicted_validation_samples.shape)
    print(measure_predictive_performance(predicted_validation_samples, val_targets))