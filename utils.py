import torch
import torch.nn as nn
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


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
