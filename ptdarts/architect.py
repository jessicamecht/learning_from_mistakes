""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
from weight_samples.visual_similarity.visual_similarity import visual_validation_similarity
from weight_samples.label_similarity.label_similarity import measure_label_similarity
from weight_samples.sample_weights import sample_weights
import torch.nn as nn
import higher
import torch.nn.functional as F


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, visual_encoder_model, coefficient_vector, w_momentum, w_weight_decay, logger=None):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net # SearchCNNController has alpha parameters and search cnn model
        self.visual_encoder_model = visual_encoder_model
        self.coefficient_vector = coefficient_vector
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.logger=logger

    def meta_learn(self, model, optimizer, input, target, input_val, target_val, coefficient_vector, visual_encoder):
        with higher.innerloop_ctx(model, optimizer) as (fmodel, foptimizer):
            logits = fmodel(input)
            weights = self.calc_instance_weights(input, target, input_val, target_val, model, coefficient_vector, visual_encoder)
            loss = torch.mean(weights * F.cross_entropy(logits, target, reduction='none'))
            foptimizer.step(loss) #replaces gradients with respect to model weights
            self.logger.info(f'Weighted training loss to update r and V: {loss}')

            logits = fmodel(input)
            val_loss = F.cross_entropy(logits, target)
            coeff_vector_gradients = torch.autograd.grad(val_loss, coefficient_vector, retain_graph=True)
            visual_encoder_gradients = torch.autograd.grad(val_loss, visual_encoder.parameters())#equivalent to backward but only for given parameters
            with torch.no_grad():
                for p, p_new in zip(self.visual_encoder_model.parameters(), visual_encoder_gradients):
                    p.copy_(p-self.w_weight_decay*p_new)
            self.coefficient_vector = self.coefficient_vector - self.w_weight_decay * coeff_vector_gradients[0]
            self.logger.info(f'New Coefficient Vector: {self.coefficient_vector}')
            self.logger.info(f'New Visual Encoder Model Weights: {self.visual_encoder_model.parameters()}')

    def calc_instance_weights(self, input_train, target_train, input_val, target_val, model, coefficient, visual_encoder):
        val_logits = model(input_val)
        crit = nn.CrossEntropyLoss(reduction='none')
        predictive_performance = crit(val_logits, target_val)
        vis_similarity = visual_validation_similarity(visual_encoder, input_val, input_train)
        label_similarity = measure_label_similarity(target_val, target_train)
        weights = sample_weights(predictive_performance, vis_similarity, label_similarity, coefficient)
        return weights

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):

        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        #calc weights
        weights = self.calc_instance_weights(trn_X, trn_y, val_X, val_y, self.net, self.coefficient_vector, self.visual_encoder_model)
        self.logger.info(f'Initial training instance weights: {weights}')
        self.virtual_step(trn_X, trn_y, xi, w_optim, weights)
        #backup
        model_backup = self.net.state_dict()
        w_optim_backup = w_optim.state_dict()

        self.meta_learn(self.net, w_optim, trn_X, trn_y, val_X, val_y, self.coefficient_vector, self.visual_encoder_model)
        #return to prev state
        self.net.load_state_dict(model_backup)
        w_optim.load_state_dict(w_optim_backup)
        # calc unrolled validation loss to update alphas
        crit = nn.CrossEntropyLoss()
        logits = self.v_net(val_X)
        loss = crit(logits, val_y) # L_val(w`)
        self.logger.info(f'Validation Loss to update Alpha: {loss}')

        # compute gradients of alpha
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        weights = self.calc_instance_weights(trn_X, trn_y, val_X, val_y, self.v_net, self.coefficient_vector,
                                             self.visual_encoder_model)
        hessian = self.compute_hessian(dw, trn_X, trn_y, weights)

        # update final alpha gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
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
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas(), retain_graph=True) # dalpha { L_trn(w+) }
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y, weights)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

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
        self.logger.info(f'Weighted training loss in Virtual Step: {loss}')
        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())
        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        # dict key is not the value, but the pointer. So original network weight have to
        # be iterated also.

        with torch.no_grad():
            for i, (w, vw, g) in enumerate(zip(self.net.weights(), self.v_net.weights(), gradients)):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))
            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def print_coefficients(self, logger):
        logger.info("####### R COEFFICIENTS #######")
        logger.info(self.coefficient_vector)
        logger.info("#####################")
    def print_visual_weights(self, logger):
        logger.info("####### VISUAL ENCODER WEIGHTS #######")
        logger.info(self.visual_encoder_model.parameters())
        logger.info("#####################")

