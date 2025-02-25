#!/usr/bin/env python
import os
import ray
import yaml
import torch

# Import the train_and_save function from your train.py file.
from train_models import train_and_save

# Wrap the imported function as a remote function.
@ray.remote
def remote_train_and_save(alpha, beta, config, device, dataset, save_path, verbose=True):
    # Simply call the original train_and_save function.
    return train_and_save(alpha, beta, config, device, dataset, save_path, verbose)

# Initialize Ray (adjust the number of CPUs if desired with num_cpus=<n>).
ray.init()

# Load configuration (using safe_load here).
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Prepare dataset and state dict directories.
current_path = os.getcwd()
path_to_datasets_folder = os.path.join(current_path, config['name_of_datasets_folder'])
path_to_save_folder = os.path.join(current_path, config['name_of_state_dicts_folder'])
os.makedirs(path_to_save_folder, exist_ok=True)

# Determine the device to use.
if torch.cuda.is_available() and config['use_gpu']:
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Prepare list to store Ray task futures.
futures = []
for alpha in config['alphas']:
    for beta in config['betas']:
        # Load the dataset for each (alpha, beta) pair once.
        dataset_path = os.path.join(path_to_datasets_folder, f"dataset_alpha{alpha}_beta{beta}.pt")
        dataset = torch.load(dataset_path, weights_only=False)
        for sample in range(config['number_of_samples']):
            save_name = f"model_alpha{alpha}_beta{beta}_sample{sample}.pt"
            save_path = os.path.join(path_to_save_folder, save_name)
            # When using GPU, divide the task into subtasks that use only one GPU.
            task = remote_train_and_save.remote(alpha, beta, config, device, dataset, save_path)
            futures.append(task)

# Wait for all tasks to complete and print their results.
results = ray.get(futures)
for result in results:
    print(result)

ray.shutdown()


