""" Architect controls architecture of cell by computing gradients of alphas """
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
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay


    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        #calc weights
        val_logits = self.net(val_X)

        r = list(self.coefficient_model.parameters())
        print(r)
        crit = nn.CrossEntropyLoss()
        u_j = crit(val_logits, val_y)
        # using W1 to calculate uj
        # 1. calculate weights
        vis_similarity = visual_validation_similarity(self.visual_encoder_model, val_X, trn_X)
        label_similarity = measure_label_similarity(val_y, trn_y)
        print(vis_similarity.shape)
        print(label_similarity.shape)
        print(r.shape)
        print(u_j.shape)

        a_i = sample_weights(u_j, vis_similarity, label_similarity, r)

        self.virtual_step(trn_X, trn_y, xi, w_optim, a_i)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y, a_i) # L_val(w`) #call weighted loss

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        r_weights = tuple(self.coefficient_model.parameters())
        visual_encoder_weights = tuple(self.visual_encoder_model.parameters())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights + visual_encoder_weights + r_weights)
        dalpha = v_grads[:len(v_alphas)]#alpha weights
        dw = v_grads[len(v_alphas):len(visual_encoder_weights)]#network weights
        d_vis_enc = v_grads[len(visual_encoder_weights):len(r_weights)]  # vis encoder weights
        d_r = v_grads[len(visual_encoder_weights):]  # vis encoder weights


        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h
            for v, dv, h in zip(self.visual_encoder_model.parameters(), d_vis_enc, hessian):
                v.grad = dv - xi*h
            for c, dr, h in zip(self.coefficient_model.parameters(), d_r, hessian):
                c.grad = dr - xi*h


    def compute_hessian(self, dw, trn_X, trn_y):
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
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def virtual_step(self, trn_X, trn_y, xi, w_optim, weights):
        """
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
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w)) #set new  weights in copy network

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)
