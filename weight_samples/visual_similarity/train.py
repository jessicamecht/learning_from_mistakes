import torch
import torch.nn as nn
from torch import optim
import os, sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader.weighted_data_loader import loadCIFARData, getWeightedDataLoaders
import weight_samples.visual_similarity.resnet_model as resnet_model
from loss import calculate_weighted_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_loader, val_loader, learning_rate=0.001, epochs=100):
    model = resnet_model.resnet50(pretrained=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    running_loss = 0

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        for steps, (inputs, labels, weights) in enumerate(train_loader):
            model.train()
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = calculate_weighted_loss(logits, labels, criterion, weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % 50 == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels, weights in val_loader:
                        inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
                    logps = model.forward(inputs)
                    batch_loss = torch.mean(criterion(logps, labels))
                    val_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Train loss: {running_loss / 50:.3f}.. "
              f"Validation loss: {val_loss / len(val_loader):.3f}.. "
              f"Validation accuracy: {accuracy / len(val_loader):.3f}")
        running_loss = 0
        model.train()

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch.save(model.state_dict(), script_dir + '/state_dicts/weighted_resnet_model.pt')
