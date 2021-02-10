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

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        running_loss = 0
        running_accuracy = 0
        for steps, (inputs, labels, weights) in enumerate(train_loader):
            model.train()
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = calculate_weighted_loss(logits, labels, criterion, weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            ps = torch.exp(logits)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            if steps % 10 == 0 and steps != 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    accuracy = 0
                    for inputs, labels, weights in val_loader:
                        inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
                        logps = model.forward(inputs)
                        batch_loss = torch.mean(criterion(logps, labels))
                        val_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                val_losses.append(val_loss / len(val_loader))
                train_losses.append(running_loss / len(train_loader))

                print(f"Epoch: {epoch}.. "
                    f"Train loss: {running_loss / steps:.3f}.. "
                    f"Train accuracy: {running_accuracy / steps:.3f}.. "
                    f"Validation loss: {val_loss / len(val_loader):.3f}.. "
                    f"Validation accuracy: {accuracy / len(val_loader):.3f}")

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch.save(model.state_dict(), script_dir + '/state_dicts/weighted_resnet_model.pt')
