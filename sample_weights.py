import torch

def sample_weights(predictive_performance, visual_similarity_scores, label_similarity_scores):
    #TODO implement coefficient vector r
    r = 1
    return torch.sigmoid(torch.dot(torch.transpose(visual_similarity_scores * label_similarity_scores * predictive_performance), r))
