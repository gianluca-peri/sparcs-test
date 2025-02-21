'''
This file contains the definition of the spectral network and the target function.
'''

import torch
import torch.nn as nn
import numpy as np
import yaml

# Import the yaml config
with open('config.yaml') as file:
    config = yaml.full_load(file)

# Definition of the spectral network (for regression)
class Phi2_Network(nn.Module):
  """
  This network expects a batch of data with rows representing single inputs
  """
  def __init__(self):
    super().__init__()

    self.input_dim = config['nn_input_dim']
    self.hidden_dim = config['nn_hidden_dim']
    self.output_dim = config['nn_output_dim']

    self.varphi1 = nn.Parameter(torch.empty(self.hidden_dim, self.input_dim), requires_grad=True)
    self.varphi2 = nn.Parameter(torch.empty(self.output_dim, self.hidden_dim), requires_grad=True)

    self.l1_diag = nn.Parameter(torch.empty(self.input_dim), requires_grad=True)
    self.l2_diag = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
    self.l3_diag = nn.Parameter(torch.empty(self.output_dim), requires_grad=True)

    self.activation = nn.ReLU()

    nn.init.xavier_uniform_(self.varphi1)
    nn.init.xavier_uniform_(self.varphi2)

    with torch.no_grad():
        self.l1_diag.fill_(0.)
        self.l2_diag.fill_(0.)
        self.l3_diag.fill_(1.)

  def forward(self, x):
    l1 = torch.diag(self.l1_diag)
    l2 = torch.diag(self.l2_diag)
    l3 = torch.diag(self.l3_diag)

    # Append 1s for bias neuron
    x = torch.cat((x, torch.ones(x.shape[0], 1, device=x.device)), dim=1)

    W_21 = torch.mm(self.varphi1, l1) - torch.mm(l2, self.varphi1)
    W_32 = torch.mm(self.varphi2, l2) - torch.mm(l3, self.varphi2)
    W_31 = torch.mm(
            torch.mm(l3, self.varphi2) - torch.mm(self.varphi2, l2),
            self.varphi1
        )

    y = torch.mm(W_31, x.t()) + torch.mm(W_32, self.activation(torch.mm(W_21, x.t())))

    return y.t()
  
# Definition of the target function
def target_function(alpha, beta, x):

    linear_term = 0.5*(1-np.tanh(beta * (alpha -0.5)))*np.mean(x, axis=1)
    non_linear_term = 0.5*(1+np.tanh(beta * (alpha -0.5)))*np.mean(x**2, axis=1)

    return linear_term + non_linear_term