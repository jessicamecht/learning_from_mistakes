""" Architect controls architecture of cell by computing gradients of alphas """
from itertools import chain
import copy
import torch
from weight_samples.visual_similarity.visual_similarity import visual_validation_similarity
from weight_samples.label_similarity.label_similarity import measure_label_similarity
from weight_samples.sample_weights import sample_weights
import torch.nn as nn

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, visual_encoder_model, coefficient_model, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net # SearchCNNController has alpha parameters and search cnn model
        self.visual_encoder_model = visual_encoder_model
        self.coefficient_model = coefficient_model
        self.v_visual_encoder_model = copy.deepcopy(visual_encoder_model)
        self.v_coefficient_model = copy.deepcopy(coefficient_model)
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay


    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):

        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        #calc weights
        # using W1 to calculate uj
        val_logits = self.net(val_X)
        r = nn.utils.parameters_to_vector(self.coefficient_model.parameters())[:-1]
        crit = nn.CrossEntropyLoss(reduction='none')
        u_j = crit(val_logits, val_y)
        # 1. calculate weights
        vis_similarity = visual_validation_similarity(self.visual_encoder_model, val_X, trn_X)
        label_similarity = measure_label_similarity(val_y, trn_y)
        a_i = sample_weights(u_j, vis_similarity, label_similarity, r)

        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim, a_i)

        val_logits = self.v_net(val_X)
        r = nn.utils.parameters_to_vector(self.v_coefficient_model.parameters())[:-1]
        crit = nn.CrossEntropyLoss(reduction='none')
        u_j = crit(val_logits, val_y)
        # using W1 to calculate uj
        # 1. calculate weights
        vis_similarity = visual_validation_similarity(self.v_visual_encoder_model, val_X, trn_X)
        label_similarity = measure_label_similarity(val_y, trn_y)
        v_ai = sample_weights(u_j, vis_similarity, label_similarity, r)


        # calc unrolled loss
        crit = nn.CrossEntropyLoss()
        logits = self.v_net(val_X)
        loss = crit(logits, val_y) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        r_weights = tuple(self.v_coefficient_model.parameters())
        visual_encoder_weights = tuple(self.v_visual_encoder_model.parameters())
        v_grads = torch.autograd.grad(loss, v_alphas  + visual_encoder_weights + r_weights + v_weights)
        dalpha = v_grads[:len(v_alphas + visual_encoder_weights + r_weights)]#alpha gradients
        dw = v_grads[len(v_alphas + visual_encoder_weights + r_weights):]#network gradients

        hessian = self.compute_hessian(dw, trn_X, trn_y, a_i)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas() + self.coefficient_model.parameters() + self.visual_encoder_model.parameters(), dalpha, hessian):
                alpha.grad = da - xi*h


    def compute_hessian(self, dw, trn_X, trn_y, weights):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """

        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y, weights)
        dalpha_pos = torch.autograd.grad(loss, chain(self.net.alphas(), self.coefficient_model.parameters(), self.visual_encoder_model.parameters())) # dalpha { L_trn(w+) }
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y, weights)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas() + self.coefficient_model.parameters() + self.visual_encoder_model.parameters()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def virtual_step(self, trn_X, trn_y, xi, w_optim, weights):
        """
        updates the weights W_2' by minimizing the weighted training loss
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        #calc weights using encoder etc and calc loss on training
        loss = self.net.loss(trn_X, trn_y, weights) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())
        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        # dict key is not the value, but the pointer. So original network weight have to
        # be iterated also.

        print('lll', self.v_net.weights())
        for i, (w, vw, g) in enumerate(zip(self.net.weights(), self.v_net.weights(), gradients)):
            m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
            vw.data = torch.clone(w - xi * (m + g + self.w_weight_decay*w))
        print(self.v_net.weights())
        # synchronize alphas
        for a, va in zip(self.net.alphas(), self.v_net.alphas()):
            va.copy_(a)

