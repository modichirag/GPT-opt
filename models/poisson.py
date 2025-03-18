import torch
import torch.nn as nn

# Define a simple linear model
class PoissonRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(PoissonRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.exp(self.linear(x)) 
    
# Define the Poisson loss function
def poisson_loss(y_true, y_pred):
    return torch.mean(y_pred - y_true * torch.log(y_pred)) 