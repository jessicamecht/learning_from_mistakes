import torch
import higher
import gc
import torch.nn.functional as F
from torchvision.datasets import CIFAR10



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
    train_loader = torch.utils.data.DataLoader(train_data, 5, shuffle=True, num_workers=0,
                                               pin_memory=True, drop_last=True)