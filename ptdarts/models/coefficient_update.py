import torch

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, criterion):
        super(LinearRegression, self).__init__()
        self.criterion = criterion
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out