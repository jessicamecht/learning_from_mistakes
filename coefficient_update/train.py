import torch
import torch.nn as nn
import os, sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coefficient_update.model import LinearRegression
from loss import calculate_weighted_loss
from weight_samples.visual_similarity.visual_similarity import visual_validation_similarity
from weight_samples.label_similarity.label_similarity import measure_label_similarity
from weight_samples.sample_weights import calculate_similarities
import ptdarts.utils as utils




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_queue, val_queue, pred_model, learning_rate, epochs, writer,config_path):
    '''Linear regression neural network which searches for the perfect coefficient vector for the training sample similiarities
    :param train_queue training data loader
    :param val_queue validation data loader '''
    logger = utils.get_logger(os.path.join(config_path, "{}.log".format('visual_embedding')))

    inputDim = next(iter(val_queue))[0].shape[0]
    outputDim = 1
    model = LinearRegression(inputDim, outputDim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    pred_model.eval()
    best_top1 = 0.
    for epoch in range(epochs):
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()
            losses = utils.AverageMeter()
            logger.info("Epoch {} LR {}".format(epoch, learning_rate))
            cur_step = (epoch + 1) * len(train_queue)
            for train_step, (train_input, train_target, _) in enumerate(train_queue):
                N = train_input.size(0)
                for val_step, loader in enumerate(val_queue):
                    val_input, val_target = loader[0].to(device), loader[1].to(device)
                    train_target = train_target.to(device)
                    train_input = train_input.to(device)
                    ######### Label Similarity
                    label_similarity = measure_label_similarity(val_target, train_target).to(device)
                    ######### Visual Similarity
                    visual_similarity = visual_validation_similarity(val_input, train_input, init=True).to(device)
                    ######### Predictive Performance
                    val_logits, _ = pred_model(val_input)
                    predictive_performance = criterion(val_logits, val_target)
                    predictive_performance = predictive_performance.to(device)
                    ######### Overall Similarity
                    similarities = calculate_similarities(predictive_performance, visual_similarity, label_similarity).to(device)
                    ######### s^Tw
                    logits_r = model(similarities)
                    ######### get weights
                    weights = torch.sigmoid(logits_r)
                    ## calculate the training loss with the training weights
                    optimizer.zero_grad()
                    logits_p, _ = pred_model(train_input)
                    cpu_weights = weights.detach()
                    loss = calculate_weighted_loss(logits_p,train_target, criterion, cpu_weights)
                    loss.backward()
                    optimizer.step()

                    prec1, prec5 = utils.accuracy(logits_p, train_target, topk=(1, 5))
                    losses.update(loss.item(), N)
                    top1.update(prec1.item(), N)
                    top5.update(prec5.item(), N)

                    if train_step % 10 == 0 and train_step != 0:
                        logger.info(
                            "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                            "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                                epoch + 1, epochs, train_step, len(train_queue) - 1, losses=losses,
                                top1=top1, top5=top5))
                    writer.add_scalar('train_coefficient_update/loss', loss.item(), cur_step)
                    writer.add_scalar('train_coefficient_update/top1', prec1.item(), cur_step)
                    writer.add_scalar('train_coefficient_update/top5', prec5.item(), cur_step)

            top1 = validate(model, criterion, val_queue, writer, cur_step, epoch, epochs, logger)
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            torch.save(model.state_dict(), script_dir + '/coefficient_update/weights/r.pt')
            # save
            if best_top1 < top1:
                best_top1 = top1
                is_best = True
            else:
                is_best = False
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            utils.save_checkpoint(state, config_path + '/resume_checkpoints')
            utils.save_checkpoint(model, config_path, is_best)

def validate(model, criterion, val_loader, writer, cur_step, epoch, epochs, logger):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (val_inputs, val_labels, val_weights) in enumerate(val_loader):
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

    writer.add_scalar('val_coefficient_update/loss', losses.avg, cur_step)
    writer.add_scalar('val_coefficient_update/top1', top1.avg, cur_step)
    writer.add_scalar('val_coefficient_update/top5', top5.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, epochs, top1.avg))

