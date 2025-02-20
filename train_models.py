import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import yaml

# Import yaml config
with open('config.yaml') as file:
  config = yaml.full_load(file)

current_path = os.getcwd()
path_to_save_folder = os.path.join(current_path, config['name_of_state_dicts_folder'])

# Set seed for reproducibility
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])

# Use graphic card if available
if torch.cuda.is_available():
  device = torch.device('cuda')
  print('Using GPU')
else:
  device = torch.device('cpu')
  print('Using CPU')

# Definition of the spectral network (for regression)
class Phi2_Network(nn.Module):
  """
  This network expects a batch of data with rows representing single inputs
  """
  def __init__(self):
    super().__init__()

    self.input_dim = 3
    self.hidden_dim = 300
    self.output_dim = 1

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

# 2D train input data generation
x = np.random.uniform(-1, 1, (10000, 2))

print(f"The shape of the input data is {x.shape}\n")

# Creation of multiple outputs for varing alpha and beta

alphas = []

for i in range(21):
  value = round(0.05*i, 3)
  alphas.append(value)

print('These are the alpha values:')
print(f'alpha={alphas}')
print(f"Total number of alphas: {len(alphas)}\n")

betas = [5, 1000]

print('These are the beta values:')
print(f'beta={betas}')
print(f"Total number of betas: {len(betas)}\n")

outputs = {}

for alpha in alphas:
  for beta in betas:
    outputs[f"output_alpha{alpha}_beta{beta}"] = target_function(alpha, beta, x)

# Create torch datasets

x_torch = torch.from_numpy(x).float()

datasets = {}

for alpha in alphas:
  for beta in betas:

    y_torch = torch.from_numpy(outputs[f'output_alpha{alpha}_beta{beta}']).float().unsqueeze(1)

    datasets[f'dataset_alpha{alpha}_beta{beta}'] = TensorDataset(x_torch, y_torch)

# Create dataloaders

dataloaders = {}

for alpha in alphas:
  for beta in betas:

    dataloader = DataLoader(datasets[f'dataset_alpha{alpha}_beta{beta}'], batch_size=100, shuffle=True)

    dataloaders[f'dataloader_alpha{alpha}_beta{beta}'] = dataloader

# Define loss function
criterion = nn.MSELoss()

# Get regularization strength
reg_strength = float(config['reg_strength'])

# Create folder to save the models
os.makedirs(path_to_save_folder, exist_ok=True)

for sample in range(config['number_of_samples']):
  for beta in betas:
    for alpha in alphas:

      print(f"Training model for alpha={alpha} and beta={beta} for sample {sample}...")

      model = Phi2_Network()
      optimizer = Adam(model.parameters(), lr=0.001)
      model.to(device)

      model.train()

      # Arbitrary high value for initial past loss
      past_global_loss = 1e6

      for epoch in range(config['epochs']):
        for inputs, targets in dataloaders[f'dataloader_alpha{alpha}_beta{beta}']:

          inputs, targets = inputs.to(device), targets.to(device)

          optimizer.zero_grad()
          pred = model(inputs)
          loss = criterion(pred, targets)

          if config['regularization']:
            for name, param in model.named_parameters():
              if 'l1' in name or 'l2' in name:
                loss += reg_strength * torch.norm(param, p=2)

          loss.backward()
          optimizer.step()

      # Save the model
      torch.save(model.state_dict(), os.path.join(path_to_save_folder, f'model_alpha{alpha}_beta{beta}_sample{sample}.pt'))