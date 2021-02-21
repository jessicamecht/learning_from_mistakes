import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_similarities(predictive_performance, visual_similarity_scores, label_similarity_scores):
    '''calculates the element wise multiplication of the three inputs
    multiplies all elements of each row element wise and
    :param predictive_performance torch of size (number val examples)
    :param visual_similarity_scores torch of size (number train examples, number val examples)
    :param label_similarity_scores torch of size (number train examples, number val examples)
    :return torch of size (number train examples, number val examples)'''
    predictive_performance = predictive_performance.T#.reshape(predictive_performance.shape[0], 1).T
    repeated_pred_perf = predictive_performance.repeat_interleave(visual_similarity_scores.shape[0], dim=0)
    assert(visual_similarity_scores.shape == label_similarity_scores.shape == repeated_pred_perf.shape)
    return visual_similarity_scores * label_similarity_scores * repeated_pred_perf

def sample_weights(predictive_performance, visual_similarity_scores, label_similarity_scores, r):
    '''performs the multiplication with coefficient vector r and squishes everything using sigmoid
    :param predictive_performance torch of size (number val examples)
    :param visual_similarity_scores torch of size (number train examples, number val examples)
    :param label_similarity_scores torch of size (number train examples, number val examples)
    :param r coefficient torch tensor of size (number val examples, 1)
    :returns tensor of size (number train examples, 1)'''
    similiarities = calculate_similarities(predictive_performance, visual_similarity_scores, label_similarity_scores)
    dp = torch.mm(similiarities, r)
    a = torch.sigmoid(dp)
    assert(a.shape[0]== visual_similarity_scores.shape[0])
    return a
