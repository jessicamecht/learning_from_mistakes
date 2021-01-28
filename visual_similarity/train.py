import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os, sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from weighted_data_loader import loadCIFARData, getWeightedDataLoaders
import train_W2
import visual_similarity.resnet_model as resnet_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 1
print_every = 2
lr = 0.003

def train(train_loader, val_loader):
    model = resnet_model.resnet50(pretrained=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    running_loss = 0

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        for steps, (inputs, labels, weights) in enumerate(train_loader):
            model.train()
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()

            loss, logps = train_W2.calculate_weighted_loss(inputs, labels, model, criterion, weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
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
              f"Train loss: {running_loss / print_every:.3f}.. "
              f"Validation loss: {val_loss / len(val_loader):.3f}.. "
              f"Validation accuracy: {accuracy / len(val_loader):.3f}")
        running_loss = 0
        model.train()

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch.save(model.state_dict(), script_dir + '/state_dicts/weighted_resnet_model.pt')

if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
    train(train_queue, val_queue)