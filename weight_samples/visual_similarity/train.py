import torch
import torch.nn as nn
from torch import optim
import os, sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import weight_samples.visual_similarity.resnet_model as resnet_model
from loss import calculate_weighted_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_loader, val_loader, learning_rate=0.001, epochs=100):
    model = resnet_model.resnet50(pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0
        for steps, (inputs, labels, weights) in enumerate(train_loader):
            model.train()
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = calculate_weighted_loss(logits, labels, criterion, weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == labels)

            if steps % 10 == 0 and steps != 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    running_val_corrects = 0.0
                    for inputs, labels, weights in val_loader:
                        inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
                        logits = model.forward(inputs)
                        loss = calculate_weighted_loss(logits, labels, criterion, weights)
                        val_loss += loss.item()
                        _, preds = torch.max(logits, 1)
                        print(preds.shape)
                        running_val_corrects += torch.sum(preds == labels)
                print(f"Epoch: {epoch}.. "
                    f"Train loss: {running_loss / steps:.3f}.. "
                    f"Train accuracy: {running_corrects / steps:.3f}.. "
                    f"Validation loss: {val_loss / len(val_loader):.3f}.. "
                    f"Validation accuracy: {running_val_corrects / len(val_loader):.3f}")

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch.save(model.state_dict(), script_dir + '/state_dicts/weighted_resnet_model.pt')
