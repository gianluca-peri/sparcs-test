import os
import ray
import yaml
import torch
from torch.utils.data import DataLoader

from train_models import train_and_save

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

current_path = os.getcwd()
path_to_datasets_folder = os.path.join(current_path, config['name_of_datasets_folder'])
path_to_save_folder = os.path.join(current_path, config['name_of_state_dicts_folder'])
os.makedirs(path_to_save_folder, exist_ok=True)

# Wrap the imported function as a remote function
# BE CAREFUL with the num_cpus and num_gpus values
if config['use_gpu']:
    @ray.remote(num_gpus=0.25)
    def remote_train_and_save(alpha, beta, learning_rate, reg_strength, epochs, device, dataloader, save_path):
            
        torch.cuda.set_per_process_memory_fraction(0.25, torch.device('cuda'))

        train_and_save(
            alpha,
            beta,
            learning_rate,
            reg_strength,
            epochs,
            device,
            dataloader,
            save_path
        )
else:
    @ray.remote(num_cpus=1)
    def remote_train_and_save(alpha, beta, learning_rate, reg_strength, epochs, device, dataloader, save_path):
        train_and_save(
            alpha,
            beta,
            learning_rate,
            reg_strength,
            epochs,
            device,
            dataloader,
            save_path
        )

# Initialize Ray
# BE CAREFUL with the num_cpus and num_gpus values
ray.init(num_cpus=16, num_gpus=1)

# Determine the device to use.
if torch.cuda.is_available() and config['use_gpu']:
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

tasks = []

for alpha in config['alphas']:
    for beta in config['betas']:
        # Load the dataset for each (alpha, beta) pair once.
        dataset_path = os.path.join(path_to_datasets_folder, f"dataset_alpha{alpha}_beta{beta}.pt")
        dataset = torch.load(dataset_path, weights_only=False)
        dataloader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True)
        for sample in range(config['number_of_samples']):
            save_name = f"model_alpha{alpha}_beta{beta}_sample{sample}.pt"
            save_path = os.path.join(path_to_save_folder, save_name)
            task = remote_train_and_save.remote(
                alpha,
                beta,
                config['learning_rate'],
                config['reg_strength'],
                config['epochs'],
                device,
                dataloader,
                save_path
            )
            tasks.append(task)

# Monitor progress
while tasks:
    completed, tasks = ray.wait(tasks)
    number_of_completed_tasks = len(completed)
    number_of_remaining_tasks = len(tasks)
    print(f"Completed {number_of_completed_tasks} tasks. {number_of_remaining_tasks} tasks remaining.")

# Shutdown Ray.
ray.shutdown()
