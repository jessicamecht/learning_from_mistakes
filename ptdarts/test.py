import torch
import higher
import gc
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from models.search_cnn import SearchCNNController
import torch.nn as nn
from models.visual_encoder import Resnet_Encoder

def meta_learn_test(model, optimizer, input, target, input_val, target_val, coefficient_vector, visual_encoder):
    with torch.backends.cudnn.flags(enabled=False):
        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=True) as (fmodel, foptimizer):
            # functional version of model allows gradient propagation through parameters of a model
            ##heavy mem allocation here
            print('memory_allocatedt1', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
                      torch.cuda.memory_reserved() / 1e9)
            logits = fmodel(input)
            print('memory_allocatedt11', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
                  torch.cuda.memory_reserved() / 1e9)
            print('memory_allocatedt2', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
                  torch.cuda.memory_reserved() / 1e9)

            weighted_training_loss = torch.mean(F.cross_entropy(logits, target, reduction='none'))
            foptimizer.step(weighted_training_loss)  # replaces gradients with respect to model weights -> w2

            logits_val = fmodel(input_val)
            meta_val_loss = F.cross_entropy(logits_val, target_val)
            coeff_vector_gradients = torch.autograd.grad(meta_val_loss, coefficient_vector, retain_graph=True)
            coeff_vector_gradients = coeff_vector_gradients[0].detach()
            visual_encoder_gradients = torch.autograd.grad(meta_val_loss,
                                                               visual_encoder.parameters())
            visual_encoder_gradients = (visual_encoder_gradients[0].detach(), visual_encoder_gradients[1].detach())# equivalent to backward for given parameters

            print('memory_allocatedtlast', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
              torch.cuda.memory_reserved() / 1e9)
            logits.detach()
            weighted_training_loss.detach()
        del logits, meta_val_loss, foptimizer, fmodel, weighted_training_loss
        gc.collect()
        torch.cuda.empty_cache()
        for module in fmodel.modules():
            del module.weight
    return visual_encoder_gradients, coeff_vector_gradients

if __name__ == "__main__":
    root = '../data'
    train_data = CIFAR10(root=root, train=True, download=True)
    test_data = CIFAR10(root=root, train=False, download=True)
    torch.manual_seed(43)
    val_data_size = len(train_data) // 2  # use half of the dataset for validation
    train_size = len(train_data) - val_data_size
    train_data = torch.utils.data.dataset.random_split(train_data, [train_size, val_data_size])
    train_loader = torch.utils.data.DataLoader(train_data, 5, shuffle=True, num_workers=1,
                                               pin_memory=True, drop_last=True)
    model = SearchCNNController(2, 16, 10, 2,
                                nn.CrossEntropyLoss(), device_ids=[0])
    w_optim = torch.optim.SGD(model.weights(), 0.01, momentum=0.01,
                              weight_decay=0.01)

    inp, targ = next(iter(train_loader))
    inp_val, targ_val = next(next(iter(train_loader)))
    inputDim = inp_val.shape[0]
    coefficient_vector  = torch.nn.Parameter(torch.ones(inputDim, 1), requires_grad=True)
    visual_encoder_model = Resnet_Encoder(nn.CrossEntropyLoss())
    meta_learn_test(model, w_optim, inp, targ, inp_val, targ_val, coefficient_vector, visual_encoder_model)
