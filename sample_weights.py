import torch

def sample_weights(predictive_performance, visual_similarity_scores, label_similarity_scores):
    #TODO implement coefficient vector r and check the equation and
    print(predictive_performance.shape, visual_similarity_scores.shape, label_similarity_scores.shape)
    return None
    r = 1
    return torch.sigmoid(torch.dot(torch.transpose(visual_similarity_scores * label_similarity_scores * predictive_performance), r))
