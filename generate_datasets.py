'''
This code generates the datasets for the project.
The datasets are saved in the format 'dataset_alpha{alpha}_beta{beta}.pt'.
'''

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import yaml

from definitions import target_function

# Import yaml config
with open('config.yaml') as file:
  config = yaml.full_load(file)

# Get path to save the models
current_path = os.getcwd()
path_to_save_folder = os.path.join(current_path, config['name_of_datasets_folder'])
os.makedirs(path_to_save_folder, exist_ok=True)

# Set seed for reproducibility
np.random.seed(config['seed'])

# 2D train input data generation
x = np.random.uniform(-1, 1, (config['number_of_dataset_points'], 2))

alphas = config['alphas']
betas = config['betas']

for alpha in alphas:
  for beta in betas:
    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(target_function(alpha, beta, x)).float().unsqueeze(1)
    dataset = TensorDataset(x_torch, y_torch)

    # Save dataset

    path_to_save_dataset = os.path.join(path_to_save_folder, f'dataset_alpha{alpha}_beta{beta}.pt')
    torch.save(dataset, path_to_save_dataset)