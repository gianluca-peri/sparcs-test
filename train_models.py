'''
This code trains the models, it is essentially the main script of the project.
The models are saved in the format 'model_alpha{alpha}_beta{beta}_sample{sample}.pt'.
'''

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import yaml

from definitions import Phi2_Network

def train_and_save(alpha, beta, learning_rate, reg_strength, epochs, device, dataloader, save_path, verbose=False):
  """
  Train the model and save it.
  """

  if verbose:
    print(f"Training model for alpha={alpha} and beta={beta}...")

  model = Phi2_Network().to(device)
  criterion = torch.nn.MSELoss().to(device)
  optimizer = Adam(model.parameters(), lr=learning_rate)

  model.train()

  for _ in range(epochs):
    for inputs, targets in dataloader:

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

  torch.save(model.state_dict(), save_path)

if __name__ == '__main__':

  # Import yaml config
  with open('config.yaml') as file:
    config = yaml.full_load(file)

  # Get path to save the models
  current_path = os.getcwd()
  path_to_datasets_folder = os.path.join(current_path, config['name_of_datasets_folder'])
  path_to_save_folder = os.path.join(current_path, config['name_of_state_dicts_folder'])
  os.makedirs(path_to_save_folder, exist_ok=True)

  # Set seed for reproducibility
  torch.manual_seed(config['seed'])

  # Use graphic card if selected and available
  if torch.cuda.is_available() and config['use_gpu']:
    device = torch.device('cuda')
    print('Using GPU')
  else:
    device = torch.device('cpu')
    print('Using CPU')

  # Get alphas and betas
  alphas = config['alphas']
  betas = config['betas']

  for alpha in alphas:
    for beta in betas:

      dataset = torch.load(
        os.path.join(path_to_datasets_folder, f'dataset_alpha{alpha}_beta{beta}.pt'),
        weights_only=False
      )

      dataloader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True)

      for sample in range(config['number_of_samples']):

        save_name = f'model_alpha{alpha}_beta{beta}_sample{sample}.pt'

        save_path = os.path.join(path_to_save_folder, save_name)

        train_and_save(
          alpha=alpha,
          beta=beta,
          learning_rate=config['learning_rate'],
          reg_strength=config['reg_strength'],
          epochs=config['epochs'],
          device=device,
          dataloader=dataloader,
          save_path=save_path,
          verbose=True
        )
                      
        print(f"Model for alpha={alpha}, beta={beta} and sample={sample} saved.")
