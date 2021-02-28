""" Architect controls architecture of cell by computing gradients of alphas and makes one step gradient descent updates of the visual encoder Vand coefficient vector r """
import copy
import torch
import torch.nn as nn
import higher
import torch.nn.functional as F
from weight_samples.sample_weights import calc_instance_weights


class Architect():
    """Object to handle the """
    def __init__(self, net, visual_encoder_model, coefficient_vector, w_momentum, w_weight_decay, eps_lr_vis_encoder, gamma_lr_coeff_vec, logger=None):
        """
        Args:
            net: current network architecture model
            visual_encoder_model: visual encoder neural network
            coefficient_vector: torch of size (number training examples, 1)
            w_momentum: weights momentum
            w_weight_decay: Int

        """
        self.net = net # SearchCNNController has alpha parameters and search cnn model
        self.visual_encoder_model = visual_encoder_model
        self.coefficient_vector = coefficient_vector
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.logger=logger
        self.eps_lr_vis_encoder = eps_lr_vis_encoder
        self.gamma_lr_coeff_vec = gamma_lr_coeff_vec

    def meta_learn(self, model, optimizer, input, target, input_val, target_val, coefficient_vector, visual_encoder):
        '''Method to meta learn the visual encoder weights and coefficient vector r, we use the higher library to be
        able to optimize through the validation loss because pytorch does not allow parameters to have grad_fn's

        Calculates the weighted training loss and performs a weight update, then calculates the validation loss and makes
        an update of the weights of the visual encoder and coefficient vector

        V' <- V - eps * d L_{Val}/dV
        r' <- r - gamma * d L_{Val}/dr

        Args:
            model: current network architecture model
            optimizer: weight optimizer for model
            input: training input of size (number of training images, channels, height, width)
            target: training target of size (number train examples, 1)
            input_val: validation input of size (number of validation images, channels, height, width)
            target_val: validation target of size (number val examples, 1)
            coefficient_vector: Tensor of size (number train examples, 1)
            visual_encoder: Visual encoder neural network to calculate instance weights
            eps: Float learning rate for visual encoder
            gamma: Float learning rate for coefficient vector
            '''

        with higher.innerloop_ctx(model, optimizer) as (fmodel, foptimizer):
            #functional version of model allows gradient propagation through parameters of a model
            logits = fmodel(input)
            weights = calc_instance_weights(input, target, input_val, target_val, model, coefficient_vector, visual_encoder)
            weighted_training_loss = torch.mean(weights * F.cross_entropy(logits, target, reduction='none'))
            foptimizer.step(weighted_training_loss) #replaces gradients with respect to model weights
            self.logger.info(f'Weighted training loss to update r and V: {weighted_training_loss}')

            logits = fmodel(input)
            meta_val_loss = F.cross_entropy(logits, target)
            coeff_vector_gradients = torch.autograd.grad(meta_val_loss, coefficient_vector, retain_graph=True)
            coeff_vector_gradients = coeff_vector_gradients[0].detach()
            visual_encoder_gradients = torch.autograd.grad(meta_val_loss, visual_encoder.parameters()) #equivalent to backward for given parameters

            #Update the visual encoder weights
            with torch.no_grad():
                for p, grad in zip(self.visual_encoder_model.parameters(), visual_encoder_gradients):
                    if p.grad is not None:
                        p.grad += grad.detach()
                    else:
                        p.grad = grad.detach()

                #Update the coefficient vector
                for p, grad in zip(self.coefficient_vector, coeff_vector_gradients):
                    if p.grad is not None:
                        p.grad += grad.detach()
                    else:
                        p.grad = grad.detach()
            #new_coefficient_vector = (self.coefficient_vector - self.gamma_lr_coeff_vec* coeff_vector_gradients)
            #self.logger.info(f'New Coefficient vector is different to old coefficient vector: {(self.coefficient_vector != new_coefficient_vector).any()}')
            #self.coefficient_vector = new_coefficient_vector
            #self.logger.info(f'New Visual Encoder Model Weights: {next(self.visual_encoder_model.parameters())}')

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):

        """ Compute unrolled loss for the alphas and backward its gradients
            Meta lerarn the coefficient vector and visual encoder with one step gradient descent
        Args:
            trn_X: (number of training images, channels, height, width)
            trn_y: (number training images, 1)
            val_X: (number of validation images, channels, height, width)
            val_y:(number validation images, 1)
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        #calc weights for weighted training loss in virtual step
        weights = calc_instance_weights(trn_X, trn_y, val_X, val_y, self.net, self.coefficient_vector, self.visual_encoder_model)
        #self.logger.info(f'Training instance weights: {weights}')
        self.virtual_step(trn_X, trn_y, xi, w_optim, weights)
        #backup before doing meta learning cause we only do one step gradient descent and don't want to change the weights just yet
        model_backup = self.net.state_dict()
        w_optim_backup = w_optim.state_dict()

        self.meta_learn(self.net, w_optim, trn_X, trn_y, val_X, val_y, self.coefficient_vector, self.visual_encoder_model)
        #return to prev state
        self.net.load_state_dict(model_backup)
        w_optim.load_state_dict(w_optim_backup)
        # calc unrolled validation loss to update alphas

        crit = nn.CrossEntropyLoss()
        logits = self.v_net(val_X)
        loss = crit(logits, val_y) # L_val(A,W2∗(W1∗(A),V,r),D(val))
        self.logger.info(f'Validation Loss to update Alpha: {loss}')

        # compute gradients of alpha
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        weights = calc_instance_weights(trn_X, trn_y, val_X, val_y, self.v_net, self.coefficient_vector,
                                             self.visual_encoder_model)
        hessian = self.compute_hessian(dw, trn_X, trn_y, weights)

        # update final alpha gradient with approximation = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h
        #print("Updated alpha gradients")


    def compute_hessian(self, dw, trn_X, trn_y, weights):
        """
        dw = dw` { L_val(A,W2∗(W1∗(A),V,r),D(val)) } with A being alpha
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { a * L_trn(w+, alpha) } - dalpha { a * L_trn(w-, alpha) }) / (2*eps) with weights a
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
        updates the weights W_2' in the virtual network by minimizing the
        weighted training loss given current state of visual encoder, coefficient vector and W1*
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
        #forward and calc weighted loss on training
        loss = self.net.loss(trn_X, trn_y, weights) # L_trn(w)
        self.logger.info(f'Weighted training loss in Virtual Step: {loss}')
        # compute gradient wrt weighted loss for network weights
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        # dict key is not the value, but the pointer. So original network weight have to
        # be iterated also.

        # Updates the weights in the virtual network
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
        logger.info(next(self.visual_encoder_model.parameters()))
        logger.info("#####################")

