import os
import ray
import yaml
import torch
import time
from torch.utils.data import DataLoader

from train_models import train_and_save

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set torch seed for reproducibility
torch.manual_seed(config['seed'])

current_path = os.getcwd()
path_to_datasets_folder = os.path.join(current_path, config['name_of_datasets_folder'])
path_to_save_folder = os.path.join(current_path, config['name_of_state_dicts_folder'])
os.makedirs(path_to_save_folder, exist_ok=True)

# Set cuda visible devices if config['use_gpu'] is True
# BE CAREFUL with the CUDA_VISIBLE_DEVICES value
if config['use_gpu']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Wrap the imported function as a remote function
# BE CAREFUL with the num_cpus and num_gpus values
if config['use_gpu']:
    @ray.remote(num_gpus=0.2)
    def remote_train_and_save(alpha, beta, learning_rate, kind_of_reg, reg_strength, epochs, device, dataloader, save_path):
        train_and_save(
            alpha,
            beta,
            learning_rate,
            kind_of_reg,
            reg_strength,
            epochs,
            device,
            dataloader,
            save_path
        )
else:
    @ray.remote(num_cpus=1)
    def remote_train_and_save(alpha, beta, learning_rate, kind_of_reg, reg_strength, epochs, device, dataloader, save_path):
        train_and_save(
            alpha,
            beta,
            learning_rate,
            kind_of_reg,
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

start_time = time.time()

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
                config['kind_of_regularization'],
                config['reg_strength'],
                config['epochs'],
                device,
                dataloader,
                save_path
            )
            tasks.append(task)

# Monitor progress
while tasks:
    _, tasks = ray.wait(tasks)
    number_of_remaining_tasks = len(tasks)
    minutes_elapsed = (time.time() - start_time) // 60
    remaining_seconds = (time.time() - start_time) % 60
    print(
        f"{number_of_remaining_tasks} tasks remaining. Time elapsed: {minutes_elapsed:.0f} minutes {remaining_seconds:.0f} seconds."
        )

# Shutdown Ray.
ray.shutdown()
