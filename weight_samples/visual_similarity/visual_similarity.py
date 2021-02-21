import torch
import torch.nn as nn
import os, sys
torch.autograd.set_detect_anomaly(True)
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader.weighted_data_loader import loadCIFARData, getWeightedDataLoaders
from weight_samples.visual_similarity import resnet_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visual_validation_similarity(model, validation_examples, training_examples, init=False):
    '''function to calculate the image similarities of training and validation examples by decoding the images
    into an embedding space
    :param validation_examples torch of size (number of val images, channels, height, width)
    :param training_examples torch of size (number of training images, channels, height, width)
    :param init: Boolean to indicate which feature extractor to use
    :returns visual_similarity torch of size (number train examples, number val examples)'''

    #create the features
    validation_embedding = extract_resnet_features(validation_examples, model) # (number val examples,number features)
    training_embedding = extract_resnet_features(training_examples, model) # (number train examples,number features)

    #dot product of each training with each validation sample V(d_tr)*V(d_val)
    x_ij_num = torch.exp(torch.mm(validation_embedding, training_embedding.T)) # (number val examples,number train examples)
    assert(x_ij_num.shape[0] == validation_embedding.shape[0] and x_ij_num.shape[1] == training_embedding.shape[0])
    x_ij_denom = torch.sum(x_ij_num, 0) # (number of train examples)
    assert(x_ij_denom.shape[0] == training_embedding.shape[0])
    visual_similarity = x_ij_num/x_ij_denom
    assert (visual_similarity.shape[0] == validation_embedding.shape[0] and visual_similarity.shape[1] == training_embedding.shape[0])
    return visual_similarity.T

def extract_resnet_features(images, model):
    '''loads a resnet pretrained model for CIFAR and gets features from the second to last layer for each image
    :param images torch of size (number images, channels, height, width)
    :param model torch model to extract features with
    :returns torch of size (number images, number features)'''
    img_var = images
    model.eval()
    with torch.no_grad():
        features_var = model(img_var) # get the output from the last hidden layer of the pretrained resnet
        print(features_var, 'features')
        features = features_var.data # get the tensor out of the variable
        print(features, 'featuresfeatures')
    return torch.squeeze(features)

'''
def create_visual_feature_extractor_model(init=False):
  #instantiates the pretrained resnet model
   #:param init Boolean chooses pretrained resnet 50 or the updated weighted resnet model
   #:returns instantiated torch nn model 

  name = 'weighted_resnet_model' if not init else 'resnet50'
  resnet_50_model = resnet_model.resnet50(pretrained=True, pretrained_name=name)
  resnet_50_model = resnet_50_model.to(device)
  modules = list(resnet_50_model.children())[:-1]
  resnet_50_model = nn.Sequential(*modules)
  for p in resnet_50_model.parameters():
    p.requires_grad = False
  return resnet_50_model
'''