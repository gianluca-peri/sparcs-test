'''
This code trains the models, it is essentially the main script of the project.
The models are saved in the format 'model_alpha{alpha}_beta{beta}_sample{sample}.pt'.
'''

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import yaml

from definitions import Phi2_Network, target_function

# Import yaml config
with open('config.yaml') as file:
  config = yaml.full_load(file)

current_path = os.getcwd()
path_to_save_folder = os.path.join(current_path, config['name_of_state_dicts_folder'])

# Set seed for reproducibility
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])

# Use graphic card if available
if torch.cuda.is_available() and config['use_gpu']:
  device = torch.device('cuda')
  print('Using GPU')
else:
  device = torch.device('cpu')
  print('Using CPU')

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
criterion = torch.nn.MSELoss()

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