import torch
import torch.nn as nn
from torch import optim
import os, sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import weight_samples.visual_similarity.resnet_model as resnet_model
from loss import calculate_weighted_loss
import ptdarts.utils as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_loader, val_loader, writer, learning_rate, epochs, config_path):
    logger = utils.get_logger(os.path.join(config_path, "{}.log".format('visual_embedding')))
    model = resnet_model.resnet50(pretrained=True)
    model = model.to(device)
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        losses = utils.AverageMeter()
        logger.info("Epoch {} LR {}".format(epoch, learning_rate))
        cur_step = (epoch + 1) * len(train_loader)
        for step, (inputs, labels, weights) in enumerate(train_loader):
            N = inputs.size(0)
            model.train()
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = calculate_weighted_loss(logits, labels, criterion, weights)
            loss.backward()
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % 10 == 0 and step != 0:
                logger.info(
                    "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, epochs, step, len(train_loader) - 1, losses=losses,
                        top1=top1, top5=top5))
            writer.add_scalar('train_visual_embedding/loss', loss.item(), cur_step)
            writer.add_scalar('train_visual_embedding/top1', prec1.item(), cur_step)
            writer.add_scalar('train_visual_embedding/top5', prec5.item(), cur_step)
        top1 = validate(model, criterion, val_loader, writer, cur_step, epoch, epochs, logger)
        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        utils.save_checkpoint(state, config_path + 'resume_checkpoints')
        utils.save_checkpoint(model, config_path, is_best)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch.save(model.state_dict(), script_dir + '/state_dicts/weighted_resnet_model.pt')

def validate(model, criterion, val_loader, writer, cur_step, epoch, epochs, logger):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()


    model.eval()
    with torch.no_grad():
        for step, val_inputs, val_labels, val_weights in enumerate(val_loader):
            N = val_inputs.size(0)
            val_inputs, val_labels, val_weights = val_inputs.to(device), val_labels.to(device), val_weights.to(device)
            logits = model.forward(val_inputs)
            loss = calculate_weighted_loss(logits, val_labels, criterion, val_weights)
            prec1, prec5 = utils.accuracy(logits, val_labels, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
            if step % 10 == 0 or step == len(val_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, epochs, step, len(val_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val_visual_embedding/loss', losses.avg, cur_step)
    writer.add_scalar('val_visual_embedding/top1', top1.avg, cur_step)
    writer.add_scalar('val_visual_embedding/top5', top5.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, epochs, top1.avg))

    return top1.avg
