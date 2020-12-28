import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_weights(predictive_performance, visual_similarity_scores, label_similarity_scores):
    label_score = label_similarity_scores.unsqueeze(2)
    pred_perf = predictive_performance.view(predictive_performance.shape[0], 1, 1, 1)
    elem_sim_mult =  visual_similarity_scores * label_score.unsqueeze(3) * pred_perf

    elem_sim_mult = torch.squeeze(elem_sim_mult, dim=3)
    transp = torch.transpose(elem_sim_mult, 1, 2)

    #TODO this is the part that takes the longest - investigate for speed improvements
    tic = time.perf_counter()
    r = torch.ones(elem_sim_mult.shape)
    r = r.to(device)
    d = torch.bmm(r, transp)
    toc = time.perf_counter()
    #print(f"third part took {toc - tic:0.4f} seconds")
    # TODO check if this is the correct dimension, it probably should be a scalar
    # TODO This is only a dummy calculation to get a scalar, there must be a mistake somewhere else before which needs to be fixed
    d_dummy = torch.sum(d, dim=1)
    d_dummy = torch.sum(d_dummy, dim=1)
    overall_similarity = torch.sigmoid(d_dummy)
    assert(overall_similarity.shape[0] == predictive_performance.shape[0])
    return overall_similarity
