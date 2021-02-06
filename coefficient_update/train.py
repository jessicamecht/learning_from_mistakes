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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_queue, val_queue, pred_model, learning_rate=0.01, epochs=100):
    '''Linear regression neural network which searches for the perfect coefficient vector for the training sample similiarities
    :param train_queue training data loader
    :param val_queue validation data loader '''
    inputDim = next(iter(val_queue))[0].shape[0]
    outputDim = 1
    model = LinearRegression(inputDim, outputDim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    pred_model.eval()

    for epoch in range(epochs):
        for step, loader in enumerate(val_queue):
            val_input, val_target = loader[0].to(device), loader[1].to(device)
            for step, (train_input, train_target, _) in enumerate(train_queue):
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

                val_loss = torch.mean(criterion(val_logits, val_target))
                print('Update Coefficients: Epoch {}, Train loss {}, Val loss {}'.format(epoch, loss.item(), val_loss))
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                torch.save(model.state_dict(), script_dir + '/coefficient_update/weights/r.pt')

