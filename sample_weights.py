import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_weights(predictive_performance, visual_similarity_scores, label_similarity_scores):
    pred_perf = predictive_performance.view(predictive_performance.shape[0], 1, 1)

    elem_sim_mult =  visual_similarity_scores * label_similarity_scores.unsqueeze(2)
    elem_sim_mult = elem_sim_mult * pred_perf

    #dimension check
    assert(elem_sim_mult.shape[0] == label_similarity_scores.shape[0])
    assert (elem_sim_mult.shape[1] == label_similarity_scores.shape[0])
    #assert(elem_sim_mult.shape[2] == visual_similarity_scores.shape[2])
    r = torch.ones(elem_sim_mult.shape)
    transp = torch.transpose(elem_sim_mult, 1, 2)

    #TODO this is the part that takes the longest - investigate for speed improvements
    tic = time.perf_counter()
    r = r.to(device)
    d = torch.bmm(transp, r)
    d = d.view(d.shape[0])
    toc = time.perf_counter()
    overall_similarity = torch.sigmoid(d)
    #dimension check
    assert(overall_similarity.shape[0] == predictive_performance.shape[0])
    return overall_similarity
