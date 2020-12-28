import torch
from torch.autograd import Variable


def sample_weights(predictive_performance, visual_similarity_scores, label_similarity_scores):
    label_score = label_similarity_scores.unsqueeze(2)
    pred_perf = predictive_performance.view(predictive_performance.shape[0], 1, 1, 1)
    elem_sim_mult =  visual_similarity_scores * label_score.unsqueeze(3) * pred_perf
    elem_sim_mult = torch.squeeze(elem_sim_mult, dim=3)
    transp = torch.transpose(elem_sim_mult, 1, 2)
    r = torch.ones(elem_sim_mult.shape)
    r = Variable(r).cuda()
    d = torch.bmm(r, transp)
    # TODO check if this is the correct dimension, it probably should be a scalar
    # TODO This is only a dummy calculation to get a scalar, there must be a mistake somewhere else before which needs to be fixed
    d_dummy = torch.sum(d, dim=1)
    d_dummy = torch.sum(d_dummy, dim=1)
    overall_similarity = torch.sigmoid(d_dummy)
    assert(overall_similarity.shape[0] == predictive_performance.shape[0])
    return overall_similarity
