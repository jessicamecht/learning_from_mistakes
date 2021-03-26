import torch
import higher
import gc
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from models.search_cnn import SearchCNNController
import torch.nn as nn
from models.visual_encoder import Resnet_Encoder
from torchvision import transforms
from weight_samples.sample_weights import calc_instance_weights


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def meta_learn_test(model, optimizer, input, target, input_val, target_val, coefficient_vector, visual_encoder):
    with torch.backends.cudnn.flags(enabled=False):
        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=True) as (fmodel, foptimizer):
            # functional version of model allows gradient propagation through parameters of a model
            ##heavy mem allocation here
            print('memory_allocatedt1', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
                      torch.cuda.memory_reserved() / 1e9)
            count_tensors(app="1")
            logits = fmodel(input)
            count_tensors(app="2")
            print('memory_allocatedt11', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
                  torch.cuda.memory_reserved() / 1e9)
            print('memory_allocatedt2', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
                  torch.cuda.memory_reserved() / 1e9)
            with torch.no_grad():
                logits_val = model(input_val)

            weights = calc_instance_weights(input, target, input_val, target_val, logits_val, coefficient_vector, visual_encoder)

            weighted_training_loss = torch.mean(weights * F.cross_entropy(logits, target, reduction='none'))
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
            count_tensors(app="3")
            logits.detach()
            weighted_training_loss.detach()
        del logits, meta_val_loss, foptimizer, fmodel, weighted_training_loss
        gc.collect()
        torch.cuda.empty_cache()
        count_tensors(app="4")
    return visual_encoder_gradients, coeff_vector_gradients

def count_tensors(app, print_new=False):
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if print_new:
                    print(type(obj), obj.size())
                count += 1
        except:
            pass
    print(str(count) + app)

if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = '../data'
    train_data = CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_data = CIFAR10(root=root, train=False, download=True, transform=transform_train)
    torch.manual_seed(43)
    val_data_size = len(train_data) // 2  # use half of the dataset for validation
    train_size = len(train_data) - val_data_size
    train_data, val_data = torch.utils.data.dataset.random_split(train_data, [train_size, val_data_size])
    train_loader = torch.utils.data.DataLoader(train_data, 5, shuffle=True, num_workers=0,
                                               pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(val_data, 5, num_workers=0,
                                               pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, 5, num_workers=0, pin_memory=True,
                                              drop_last=True)
    model = SearchCNNController(3, 16, 10, 8,
                                nn.CrossEntropyLoss(), device_ids=[0])


    inp, targ = next(iter(train_loader))
    inp_val, targ_val = next(iter(train_loader))
    inp, targ = inp.to(device), targ.to(device)
    inp_val, targ_val = inp_val.to(device), targ_val.to(device)
    model = model.to(device)
    inputDim = inp_val.shape[0]
    coefficient_vector  = torch.ones(inputDim, 1, requires_grad=True, device=device)
    coefficient_vector = coefficient_vector.to(device)
    visual_encoder_model = Resnet_Encoder(nn.CrossEntropyLoss())
    visual_encoder_model = visual_encoder_model.to(device)
    w_optim = torch.optim.SGD(list(model.parameters()), 0.01)
    a,b = meta_learn_test(model, w_optim, inp, targ, inp_val, targ_val, coefficient_vector, visual_encoder_model)
    count_tensors(app="5")

    del inp, inp_val, targ, targ_val, model, visual_encoder_model, coefficient_vector, a, b, w_optim, test_data, test_loader, val_data, valid_loader, root, train_data, train_size
    gc.collect()
    torch.cuda.empty_cache()

    count_tensors(app="6", print=True)
    print('memory_allocatedt2klhljkh', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
          torch.cuda.memory_reserved() / 1e9)
