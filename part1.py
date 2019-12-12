import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class FeedForward(torch.nn.Module):
    
  def __init__(self, input_size, output_size):
    super(FeedForward, self).__init__()
    self.hidden = nn.Linear(input_size, 1) #One hidden layer
    self.out = nn.Linear(hidden_size,output_size)
    
  def forward(self, x):
    x = self.hidden(x)
    x = F.tanh(x)
    x = self.out(x)
    return x

network = FeedForward(input_size=2, hidden_size = 1, output_size=2)
optimizer = torch.optim.SGD(network.parameters(), lr=0.02)
loss_function = torch.nn.CrossEntropyLoss() #Loss function