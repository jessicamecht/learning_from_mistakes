import torch

def sample_weights(predictive_performance, visual_similarity_scores, label_similarity_scores):
    label_score = label_similarity_scores.unsqueeze(2)
    pred_perf = predictive_performance.view(predictive_performance.shape[0], 1, 1, 1)
    elem_sim_mult =  visual_similarity_scores * label_score.unsqueeze(3) * pred_perf
    elem_sim_mult = torch.squeeze(elem_sim_mult, dim=3)
    transp = torch.transpose(elem_sim_mult, 1, 2)
    r = torch.ones(elem_sim_mult.shape)
    d = torch.bmm(r, transp)
    overall_similarity = torch.sigmoid(d)
    #TODO check if this is the correct dimention, it probably should be a scalar
    assert(overall_similarity.shape[0] == predictive_performance.shape[0])
    return overall_similarity
