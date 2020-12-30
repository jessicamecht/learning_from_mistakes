import torch
import torch.nn as nn
import os, sys
from visual_similarity import resnet_model
torch.autograd.set_detect_anomaly(True)
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weighted_data_loader import loadCIFARData, getWeightedDataLoaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visual_validation_similarity(validation_examples, training_examples, model):
    '''function to calculate the image similarities by decoding the images
    into an embedding space using an autoencoder neural network'''
    validation_examples_embedding = extract_resnet_features(validation_examples, model)
    training_examples_embedding = extract_resnet_features(training_examples, model)
    #x_ij_num = torch.exp(training_examples_embedding.unsqueeze(1) * validation_examples_embedding)
    #x_ijh_denom = torch.sum(
    #    torch.exp(training_examples_embedding.expand_as(validation_examples_embedding).unsqueeze(1) * validation_examples_embedding),
    #    dim=0)
    t, v = torch.squeeze(torch.transpose(training_examples_embedding, 1, 2), dim=3), torch.squeeze(validation_examples_embedding, dim=3)

    x_ij_num = torch.exp(torch.bmm(t,v))

    expanded_t = training_examples_embedding.unsqueeze(1).repeat(1, 256, 1, 1, 1)

    expanded_t = torch.squeeze(expanded_t, dim=3)
    validation_examples_embedding = torch.squeeze(validation_examples_embedding, dim=3)
    dot_products = torch.empty(validation_examples.shape[0],1,1).to(device)
    for elem in expanded_t:
        dot_product = torch.bmm(elem.view(256, 1, 2048), validation_examples_embedding.view(256, 2048, 1))
        dot_products = torch.cat((dot_products, dot_product), dim=1)

    x_ijh_denom = torch.sum(
        torch.exp(dot_products),
        dim=1)
    similarity = x_ij_num / x_ijh_denom
    assert(not torch.isnan(similarity).any())
    #dimension check
    #assert(similarity.shape[0] == training_examples_embedding.shape[0])
    #assert(similarity.shape[1] == validation_examples_embedding.shape[0])
    #assert(similarity.shape[2] == validation_examples_embedding.shape[1])
    return similarity

def extract_resnet_features(images, model):
    '''loads a resnet pretrained model for CIFAR and gets features from the second to last layer for each image'''
    img_var = images
    model.eval()
    with torch.no_grad():
        features_var = model(img_var) # get the output from the last hidden layer of the pretrained resnet
        features = features_var.data # get the tensor out of the variable
    return features


if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_loader, val_loader, test_loader = getWeightedDataLoaders(train_data, val_data, test_data, batch_size=10)
    train_imgs, train_targets, train_weights = next(iter(train_loader))
    val_imgs, val_targets, val_weights = next(iter(val_loader))

    print(extract_resnet_features(train_imgs).shape)
    print(visual_validation_similarity(train_imgs, val_imgs).shape)
