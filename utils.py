from visual_similarity.visual_similarity import resnet_model
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def progress(batch_idx, len_epoch, data_loader):
    '''return the progress that can be used for training or longer calculations'''
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(data_loader, 'n_samples'):
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
    else:
        current = batch_idx
        total = len_epoch
    return base.format(current, total, 100.0 * current / total)

def create_visual_feature_extractor_model(init=False):
  name = 'weighted_resnet_model' if not init else 'resnet50'
  resnet_50_model = resnet_model.resnet50(pretrained=True, pretrained_name=name)
  resnet_50_model = resnet_50_model.to(device)
  modules = list(resnet_50_model.children())[:-1]
  resnet_50_model = nn.Sequential(*modules)
  for p in resnet_50_model.parameters():
    p.requires_grad = False
  return resnet_50_model