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

    net_crit = calculate_weighted_loss
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)


    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)


    # Init Visual encoder model
    visual_encoder_model = Resnet_Encoder(nn.CrossEntropyLoss())
    visual_encoder_model = visual_encoder_model.to(device)
    inputDim = next(iter(valid_loader))[0].shape[0]
    coeff_vector = torch.ones(inputDim, 1, requires_grad=True).to(device)

    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999), weight_decay=config.alpha_weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, visual_encoder_model, coeff_vector, config.w_momentum, config.w_weight_decay, logger)


    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)
        architect.print_coefficients(logger)
        architect.print_visual_weights(logger)

        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, writer, logger)

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


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, writer, logger):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train_search/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y, weights_train), (val_X, val_y, weights_valid)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()

        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim) #calculates gradient for alphas and updates V and r
        alpha_optim.step() #updates weights for alphas
        #print("Updated alphas")

        # phase 1. child network step (w) minimizes the training loss
        w_optim.zero_grad()
        logits = model(trn_X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()
        #print("Updated W1 weights with training CEL ")

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1
        del trn_y, trn_X, val_y, val_X, weights_train, weights_valid
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
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
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

    return top1.avg


if __name__ == "__main__":
    main()
