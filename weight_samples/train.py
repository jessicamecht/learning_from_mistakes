import numpy as np
import torch
import os,sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weighted_data_loader import loadCIFARData, getWeightedDataLoaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lr = 0.01
epochs = 100
print_every = 10
inputSize = 3*32*32

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

def train(train_loader, valid_loader):
    model = LinearRegression(inputSize, output_size)
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        for steps, (inputs, labels, weights) in enumerate(train_loader):
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            print(loss)
            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels, weights in valid_loader:
                        inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
                    logps = model.forward(inputs)
                    batch_loss = torch.mean(criterion(logps, labels))
                    val_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print('epoch {}, loss {}'.format(epoch, loss.item()))

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch.save(model.state_dict(), script_dir + '/state_dicts/weighted_resnet_model.pt')

if __name__ == "__main__":
    train_data, val_data, test_data = loadCIFARData()
    train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
    train(train_queue, val_queue)