import torch
import torch.nn as nn
import os, sys
from visual_similarity import resnet_model
from torch.autograd import Variable
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weighted_data_loader import loadCIFARData, getWeightedDataLoaders

def visual_validation_similarity(validation_examples, training_examples):
    '''function to calculate the image similarities by decoding the images
    into an embedding space using an autoencoder neural network'''
    print(validation_examples.shape, training_examples.shape)
    validation_examples_embedding = extract_resnet_features(validation_examples)
    training_examples_embedding = extract_resnet_features(training_examples)
    x_ij_num = torch.exp(validation_examples_embedding.unsqueeze(1) * training_examples_embedding)
    x_ijh_denom = torch.sum(
        torch.exp(training_examples_embedding.expand_as(validation_examples_embedding).unsqueeze(1) * validation_examples_embedding),
        dim=0)
    similarity = x_ij_num / x_ijh_denom
    similarity = torch.squeeze(similarity, dim=3)
    return similarity

def extract_resnet_features(images):
    '''loads a resnet pretrained model for CIFAR and gets features from the second to last layer for each image'''
    resnet_18_model = resnet_model.resnet50(pretrained=True)
    modules=list(resnet_18_model.children())[:-1]
    resnet_18_model =nn.Sequential(*modules)
    for p in resnet_18_model.parameters():
        p.requires_grad = False
    img_var = Variable(images)
    features_var = resnet_18_model(img_var) # get the output from the last hidden layer of the pretrained resnet
    features = features_var.data # get the tensor out of the variable
    return features


if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_loader, val_loader, test_loader = getWeightedDataLoaders(train_data, val_data, test_data, batch_size=10)
    train_imgs, train_targets, train_weights = next(iter(train_loader))
    val_imgs, val_targets, val_weights = next(iter(val_loader))
    val_imgs = val_imgs.permute(0, 3, 1, 2).float() / 255.
    train_imgs = train_imgs.permute(0, 3, 1, 2).float() / 255.

    print(extract_resnet_features(train_imgs).shape)
    print(visual_validation_similarity(train_imgs, val_imgs).shape)
