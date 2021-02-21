""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from ptdarts.config import SearchConfig
import ptdarts.utils as utils
from ptdarts.models.search_cnn import SearchCNNController
from ptdarts.architect import Architect
import gc
from loss import calculate_weighted_loss
from ptdarts.models.visual_encoder import Resnet_Encoder
from weight_samples.visual_similarity.visual_similarity import visual_validation_similarity
from weight_samples.label_similarity.label_similarity import measure_label_similarity
from ptdarts.models.coefficient_update import LinearRegression
from weight_samples.sample_weights import sample_weights
import copy

config = SearchConfig()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(train_loader, valid_loader, config_path, writer):
    writer.add_text('config', config.as_markdown(), 0)

    logger = utils.get_logger(os.path.join(config_path, "{}.log".format(config.name)))
    config.print_params(logger.info)
    logger.info("Logger is set - training start")

    # set default gpu device id

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_channels = 3
    n_classes = 10

    net_crit = nn.CrossEntropyLoss(reduction='none').to(device)
    net_crit = lambda logits, target, weights: calculate_weighted_loss(logits, target, net_crit, weights).to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)


    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # Init Visual encoder model
    visual_encoder_model = Resnet_Encoder(nn.CrossEntropyLoss())
    visual_encoder_model = visual_encoder_model.to(device)
    visual_encoder_optimizer = torch.optim.Adam(visual_encoder_model.parameters(), config.alpha_lr, betas=(0.5, 0.999),
                                                weight_decay=config.alpha_weight_decay)

    # Init coefficient vector r update model
    inputDim = next(iter(valid_loader))[0].shape[0]
    coefficient_update_model = LinearRegression(inputDim, 1, nn.CrossEntropyLoss())
    coefficient_update_model = coefficient_update_model.to(device)
    coefficient_update_optimizer = torch.optim.Adam(coefficient_update_model.parameters(), config.alpha_lr,
                                                    betas=(0.5, 0.999),
                                                    weight_decay=config.alpha_weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, visual_encoder_model, coefficient_update_model, config.w_momentum, config.w_weight_decay)


    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, visual_encoder_model, coefficient_update_model, visual_encoder_optimizer, coefficient_update_optimizer, lr, epoch, writer, logger)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step, writer, logger)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        #plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        #plot(genotype.normal, plot_path + "-normal", caption)
        #plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            genotype_txt = open("genotype.txt", "w")
            n = genotype_txt.write(best_genotype)
            genotype_txt.close()

            is_best = True
        else:
            is_best = False
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                 'w_optim': w_optim.state_dict(), "alpha_optim": alpha_optim.state_dict()}
        utils.save_checkpoint(state, config_path + '/resume_checkpoints')
        utils.save_checkpoint(model, config_path, is_best)

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    writer.add_graph(model.encoder, next(iter(valid_loader)))
    return model


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, v_model, c_model, v_optim, c_optim, lr, epoch, writer, logger):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    naive_top1 = utils.AverageMeter()
    naive_top5 = utils.AverageMeter()
    naive_losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train_search/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y, weights_train), (val_X, val_y, weights_valid)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y, weights_train = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True), weights_train.to(device, non_blocking=True)
        val_X, val_y, weights_valid = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True), weights_valid.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        #get the weights for the current
        alpha_optim.zero_grad()
        v_optim.zero_grad()
        c_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)#calculates gradient for alphas, visual encoder and coefficient model
        alpha_optim.step()
        v_optim.step()
        c_optim.step()

        r = c_model.parameters()

        # phase 1. child network step (w) minimizes the training loss
        w_optim.zero_grad()
        logits = model(trn_X)
        loss_naive = model.criterion(logits, trn_y)
        loss_naive.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()
        prec1_naive, prec5_naive = utils.accuracy(logits, trn_y, topk=(1, 5))

        #Validation network W2 apply W1∗(A) to the validation dataset D(val) and see how it performs on the validation examples
        val_logits = model(val_X)
        u_j = model.criterion(val_logits, val_y)
        #using W1 to calculate uj
        #1. calculate weights
        vis_similarity = visual_validation_similarity(v_model, val_X, trn_X)
        label_similarity = measure_label_similarity(val_y, trn_y)
        a_i = sample_weights(u_j, vis_similarity, label_similarity, r)

        #2. weighted training loss -> W2 network weights for neural architecture search
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = calculate_weighted_loss(logits, trn_y, nn.CrossEntropyLoss(reduction='none'), a_i)
        loss.backward()
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()
        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))



        del trn_X, trn_y, weights_train, val_X, val_y, weights_valid
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        naive_losses.update(loss_naive.item(), N)
        naive_top1.update(prec1_naive.item(), N)
        naive_top5.update(prec5_naive.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
            logger.info(
                "Train naive : [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=naive_losses,
                    top1=naive_top1, top5=naive_top5))

        writer.add_scalar('train_search/loss', loss.item(), cur_step)
        writer.add_scalar('train_search/top1', prec1.item(), cur_step)
        writer.add_scalar('train_search/top5', prec5.item(), cur_step)

        writer.add_scalar('train_search_naive/loss', loss_naive.item(), cur_step)
        writer.add_scalar('train_search_naive/top1', prec1_naive.item(), cur_step)
        writer.add_scalar('train_search_naive/top5', prec5_naive.item(), cur_step)
        cur_step += 1
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step, writer, logger):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()


    model.eval()

    with torch.no_grad():
        for step, (X, y, weights) in enumerate(valid_loader):
            X, y, weights = X.to(device, non_blocking=True), y.to(device, non_blocking=True), weights.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))
            del X, y, weights
            gc.collect()
            torch.cuda.empty_cache()

    writer.add_scalar('val_search/loss', losses.avg, cur_step)
    writer.add_scalar('val_search/top1', top1.avg, cur_step)
    writer.add_scalar('val_search/top5', top5.avg, cur_step)

7
    return top1.avg


if __name__ == "__main__":
    main()
