import os
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

# Import yaml config

with open('config.yaml') as file:
  config = yaml.full_load(file)

current_path = os.getcwd()
path_to_state_dict_folder= os.path.join(current_path, config['name_of_state_dicts_folder'])



NUMBER_OF_SAMPLES = 100

# Create the save folder if not present

os.makedirs(PATH_TO_SHORTCUT_SAVE_FOLDER, exist_ok=True)

# Definition of the network

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
    x = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)

    W_21 = torch.mm(self.varphi1, l1) - torch.mm(l2, self.varphi1)
    W_32 = torch.mm(self.varphi2, l2) - torch.mm(l3, self.varphi2)
    W_31 = torch.mm(
            torch.mm(l3, self.varphi2) - torch.mm(self.varphi2, l2),
            self.varphi1
        )

    y = torch.mm(W_31, x.t()) + torch.mm(W_32, self.activation(torch.mm(W_21, x.t())))

    return y.t()

models = {}

# Definition of the target function

def target_function(alpha, beta, x):

    linear_term = 0.5*(1-np.tanh(beta * (alpha -0.5)))*np.mean(x, axis=1)
    non_linear_term = 0.5*(1+np.tanh(beta * (alpha -0.5)))*np.mean(x**2, axis=1)

    return linear_term + non_linear_term

# x creation

x = np.random.uniform(-1, 1, (10000, 2))

# Selection of the alpha and beta values

alphas = []

for i in range(21):
  value = round(0.05*i, 3)
  alphas.append(value)

print('These are the alpha values:')
print(f'alpha={alphas}')
print(f"Total number of alphas: {len(alphas)}\n")

betas = [5, 1000]

# Creation of the outputs

outputs = {}

for alpha in alphas:
  for beta in betas:
    outputs[f"output_alpha{alpha}_beta{beta}"] = target_function(alpha, beta, x)

# Loading the models

models = {}

for beta in betas:
  for alpha in alphas:
    for sample in range(NUMBER_OF_SAMPLES):
      model = Phi2_Network()
      model.load_state_dict(
        torch.load(
          os.path.join(
            path_to_state_dict_folder,
            f'model_alpha{alpha}_beta{beta}_sample{sample}.pt'
          )
        )
      models[f'model_alpha{alpha}_beta{beta}_sample{sample}'] = model

# Plot 3D graphs of y and pred y to show the goodness of the fit

for sample in tqdm(range(NUMBER_OF_SAMPLES), desc='Making goodnes of fit plots'):
  for beta in betas:

    # Make multiplot figure
    num_cols = 7
    num_rows = int(np.ceil(len(alphas) / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, subplot_kw={'projection': '3d'}, figsize=(20, 10))
    axs = axs.flatten()

    fig.suptitle(f'Results for $\\beta={beta}$, sample {sample}')

    for i, alpha in enumerate(alphas):

      model = models[f'model_alpha{alpha}_beta{beta}_sample0']

      # Get the predictions
      model.eval()
      y_pred = model(torch.from_numpy(x).float()).detach().numpy()

      axs[i].set_title(f'$\\alpha={alpha}$')
      axs[i].set_xlabel('$x_1$')
      axs[i].set_ylabel('$x_2$')
      axs[i].set_zlabel('$y$')

      # Plot the predictions
      axs[i].scatter(x[:, 0], x[:, 1], y_pred, c='red', label='prediction')

      abs_error = []
      for j in range(len(outputs[f'output_alpha{alpha}_beta{beta}'])):
        abs_error.append(abs(outputs[f'output_alpha{alpha}_beta{beta}'][j] - y_pred[j]))

      # Plot the abs error
      axs[i].scatter(x[:, 0], x[:, 1], abs_error, c='green', label='abs error')

    # Make global legend
    fig.legend(
        handles = axs[0].get_legend_handles_labels()[0],
        labels = axs[0].get_legend_handles_labels()[1],
        loc='lower center'
        )

    os.makedirs(f'{PATH_TO_SHORTCUT_SAVE_FOLDER}fits', exist_ok=True)

    plt.savefig(f'{PATH_TO_SHORTCUT_SAVE_FOLDER}fits/fit_beta{beta}_sample{sample}.png')

# Make mean error graph

for beta in tqdm(betas, desc='Making mean error plots'):

  mean_error = []
  std_mean_error = []

  fig = plt.figure()
  ax = fig.add_subplot()
  ax.set_title('$\\alpha$ vs. mean error, $\\beta={beta}$')
  ax.set_xlabel('$\\alpha$')
  ax.set_ylabel('mean error')

  ax.axhline(y=0, color='grey', linestyle='--')

  for alpha in alphas:

    errors_mean_on_samples = []

    for sample in range(NUMBER_OF_SAMPLES):
      model = models[f'model_alpha{alpha}_beta{beta}_sample{sample}']

      # Get the predictions
      model.eval()
      y_pred = model(torch.from_numpy(x).float()).detach().numpy()

      abs_error = []

      for j in range(len(outputs[f'output_alpha{alpha}_beta{beta}'])):
        abs_error.append(abs(outputs[f'output_alpha{alpha}_beta{beta}'][j] - y_pred[j]))

      errors_mean_on_samples.append(np.mean(abs_error))

    mean_error.append(np.mean(errors_mean_on_samples))
    std_mean_error.append(np.std(errors_mean_on_samples)/np.sqrt(NUMBER_OF_SAMPLES))

  # Normalization
  max_value = max(mean_error)

  for elem in mean_error:
    elem = elem/max_value

  for elem in std_mean_error:
    elem = elem/max_value

  # Plotting
  ax.errorbar(alphas,
              mean_error,
              yerr=std_mean_error,
              label=f'$\\beta={beta}$',
              marker='o',
              markersize=5,
              capsize=5,
              capthick=1)

  plt.legend()

  plt.savefig(f'{PATH_TO_SHORTCUT_SAVE_FOLDER}mean_error_beta{beta}.png')

# Make graph of mean of L_1 and L_2

mean_l1_l2 = {}
mean_std_l1_l2 = {}

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('$\\alpha$ vs. mean of concatenation of eigenvalues')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('mean of concatenation')

ax.axhline(y=0, color='grey', linestyle='--')

for beta in tqdm(betas, desc='Making mean of L1 and L2 plots'):

  color = np.random.rand(3,)

  for alpha in alphas:

    means = []
    for sample in range(NUMBER_OF_SAMPLES):
      l1_diag = models[f'model_alpha{alpha}_beta{beta}_sample{sample}'].l1_diag
      l2_diag = models[f'model_alpha{alpha}_beta{beta}_sample{sample}'].l2_diag
      means.append(torch.mean(torch.abs(torch.cat((l1_diag, l2_diag)))).detach().numpy())

    mean_l1_l2[alpha] = np.mean(means)
    mean_std_l1_l2[alpha] = np.std(means)/np.sqrt(NUMBER_OF_SAMPLES)

  # Normalization
  max_value = max(mean_l1_l2.values())

  for key, value in mean_l1_l2.items():
    mean_l1_l2[key] = value/max_value

  for key, value in mean_std_l1_l2.items():
    mean_std_l1_l2[key] = value/max_value

  # Plotting
  ax.errorbar(alphas,
              list(mean_l1_l2.values()),
              yerr=list(mean_std_l1_l2.values()),
              label=f'$\\beta={beta}$',
              color=color,
              marker='o',
              markersize=5,
              capsize=5,
              capthick=1)

ax.legend()

plt.savefig(f'{PATH_TO_SHORTCUT_SAVE_FOLDER}mean_l1_l2.png')

# Defining function to calculate W_21, W_32, and W_31 from varphi1, varphi2, l1_diag, l2_diag, l3_diag

def calculate_Ws(varphi1, varphi2, l1_diag, l2_diag, l3_diag):

  l1 = torch.diag(l1_diag)
  l2 = torch.diag(l2_diag)
  l3 = torch.diag(l3_diag)
  W_21 = torch.mm(varphi1, l1) - torch.mm(l2, varphi1)
  W_32 = torch.mm(varphi2, l2) - torch.mm(l3, varphi2)
  W_31 = torch.mm(
          torch.mm(l3, varphi2) - torch.mm(varphi2, l2),
          varphi1
      )

  W_21 = W_21.detach().numpy()
  W_32 = W_32.detach().numpy()
  W_31 = W_31.detach().numpy()

  return {'W_21': W_21, 'W_32': W_32, 'W_31': W_31}

# Defining function to calculate N

def calculate_N(W_21, W_32):

  N = np.zeros((W_32.shape[0], W_32.shape[1], W_21.shape[1]))

  for i in range(W_32.shape[0]):
    for j in range(W_32.shape[1]):
      for k in range(W_21.shape[1]):
        N[i, j, k] = W_32[i,j]*W_21[j,k]

  return N

Ns = {}

for sample in tqdm(range(NUMBER_OF_SAMPLES), desc='Calculating Ns'):
  for beta in betas:
    for alpha in alphas:

      model = models[f'model_alpha{alpha}_beta{beta}_sample{sample}']

      Ws = calculate_Ws(model.varphi1, model.varphi2, model.l1_diag, model.l2_diag, model.l3_diag)

      N = calculate_N(Ws['W_21'], Ws['W_32'])

      Ns[f'N_alpha{alpha}_beta{beta}_sample{sample}'] = N

fig = plt.figure()
ax = fig.add_subplot()

ax.set_title(f'$\\alpha$ vs. norm of N')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$||N||$')

ax.axhline(y=0, color='grey', linestyle='--')

for beta in tqdm(betas, desc='Making norm of N plots'):

  # One element for each alpha
  # The mean is made on the samples
  mean_of_norm_of_N = []
  std_mean = []

  for alpha in alphas:

    norms_samples = []
    for sample in range(NUMBER_OF_SAMPLES):
      norms_samples.append(np.linalg.norm(Ns[f'N_alpha{alpha}_beta{beta}_sample{sample}']))

    mean_of_norm_of_N.append(np.mean(norms_samples))
    std_mean.append(np.std(norms_samples)/np.sqrt(NUMBER_OF_SAMPLES))

    # Normalization

    max_value = max(mean_of_norm_of_N)

    for elem in mean_of_norm_of_N:
      elem = elem/max_value

    for elem in std_mean:
      elem = elem/max_value

  # Plotting
  ax.errorbar(alphas,
              mean_of_norm_of_N,
              yerr=std_mean,
              label=f'$\\beta={beta}$',
              marker='o',
              markersize=5,
              capsize=5,
              capthick=1)

ax.legend()

plt.savefig(f'{PATH_TO_SHORTCUT_SAVE_FOLDER}norm_of_N.png')

# PLOTTING THE NEURAL GRAPH IN 3D

# Definition of the helper functions

def plot_neurons(ax, input_dim, hidden_dim, output_dim):

  # First layer position, centered at zero
  shift = input_dim/2

  first_layer_x_positions = [(i-shift)*60 for i in range(input_dim)]
  first_layer_y_positions = [0 for i in range(input_dim)]
  first_layer_z_positions = [0 for i in range(input_dim)]

  # Second layer position

  shift = hidden_dim/2

  second_layer_x_positions = [i-shift for i in range(hidden_dim)]
  second_layer_y_positions = [1 for i in range(hidden_dim)]
  second_layer_z_positions = [0 for i in range(hidden_dim)]

  # Third layer positions

  shift = output_dim/2

  third_layer_x_positions = [i-shift for i in range(output_dim)]
  third_layer_y_positions = [2 for i in range(output_dim)]
  third_layer_z_positions = [0 for i in range(output_dim)]

  # Plotting

  ax.scatter(first_layer_x_positions, first_layer_y_positions, first_layer_z_positions, c='black')
  ax.scatter(second_layer_x_positions, second_layer_y_positions, second_layer_z_positions, c='black')
  ax.scatter(third_layer_x_positions, third_layer_y_positions, third_layer_z_positions, c='black')

  return {'first_layer_x_positions': first_layer_x_positions,
          'first_layer_y_positions': first_layer_y_positions,
          'first_layer_z_positions': first_layer_z_positions,
          'second_layer_x_positions': second_layer_x_positions,
          'second_layer_y_positions': second_layer_y_positions,
          'second_layer_z_positions': second_layer_z_positions,
          'third_layer_x_positions': third_layer_x_positions,
          'third_layer_y_positions': third_layer_y_positions,
          'third_layer_z_positions': third_layer_z_positions}

def mean_matrix_on_samples(matrices):

  mean_matrix = np.zeros(matrices[0].shape)

  for matrix in matrices:
    mean_matrix += matrix

  mean_matrix /= len(matrices)

  return mean_matrix

def quantization(matrix, treshold):

  if matrix.ndim == 2:
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        if abs(matrix[i,j]) < treshold:
          matrix[i,j] = 0
        else:
          matrix[i,j] = 1
  else:
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        for k in range(matrix.shape[2]):
          if abs(matrix[i,j,k]) < treshold:
            matrix[i,j,k] = 0
          else:
            matrix[i,j,k] = 1

  return matrix

def plot_connections(ax, N, W_31, positions_dict):

  # Plot the connections passing through the hidden layer

  for i in range(N.shape[0]):
    for j in range(N.shape[1]):
      for k in range(N.shape[2]):
        if N[i,j,k] != 0:

          ax.plot([positions_dict['third_layer_x_positions'][i],
                   positions_dict['second_layer_x_positions'][j],
                   positions_dict['first_layer_x_positions'][k]],
                  [positions_dict['third_layer_y_positions'][i],
                   positions_dict['second_layer_y_positions'][j],
                   positions_dict['first_layer_y_positions'][k]],
                  [positions_dict['third_layer_z_positions'][i],
                   positions_dict['second_layer_z_positions'][j],
                   positions_dict['first_layer_z_positions'][k]],
                  color='violet')


  # Plot skip layer connections between the input layer and the output layer, arching over in 3D

  for i in range(W_31.shape[0]):
    for j in range(W_31.shape[1]):
      if W_31[i,j] != 0:

        mid_point_in_x_space = (positions_dict['first_layer_x_positions'][j] + positions_dict['third_layer_x_positions'][i])/2
        mid_point_in_y_space = (positions_dict['first_layer_y_positions'][j] + positions_dict['third_layer_y_positions'][i])/2
        mid_point_in_z_space = 1

        ax.plot([positions_dict['first_layer_x_positions'][j], mid_point_in_x_space],
                [positions_dict['first_layer_y_positions'][j], mid_point_in_y_space],
                [positions_dict['first_layer_z_positions'][j], mid_point_in_z_space],
                color='blue')

        ax.plot([mid_point_in_x_space, positions_dict['third_layer_x_positions'][i]],
                [mid_point_in_y_space, positions_dict['third_layer_y_positions'][i]],
                [mid_point_in_z_space, positions_dict['third_layer_z_positions'][i]],
                color='blue')

# Start of the plot

TRESHOLD = 1e-2

for sample in tqdm(range(NUMBER_OF_SAMPLES), desc='Making architectures plots'):
  for beta in betas:

    # Make multiplot figure
    num_cols = 7
    num_rows = int(np.ceil(len(alphas) / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, subplot_kw={'projection': '3d'}, figsize=(20, 10))
    axs = axs.flatten()

    fig.suptitle(f'Results for $\\beta={beta}$')

    for j, alpha in enumerate(alphas):

      model = models[f'model_alpha{alpha}_beta{beta}_sample{sample}']

      varphi1 = model.varphi1
      varphi2 = model.varphi2
      l1_diag = model.l1_diag
      l2_diag = model.l2_diag
      l3_diag = model.l3_diag

      W_21 = calculate_Ws(varphi1, varphi2, l1_diag, l2_diag, l3_diag)['W_21']
      W_32 = calculate_Ws(varphi1, varphi2, l1_diag, l2_diag, l3_diag)['W_32']
      W_31 = calculate_Ws(varphi1, varphi2, l1_diag, l2_diag, l3_diag)['W_31']

      N = calculate_N(W_21, W_32)

      axs[j].set_title(f'$\\alpha={alpha}$')
      axs[j].set_axis_off()

      input_dim = model.input_dim
      hidden_dim = model.hidden_dim
      output_dim = model.output_dim

      positions_dict = plot_neurons(axs[j], input_dim, hidden_dim, output_dim)

      N_quantized = quantization(N, TRESHOLD)
      W_31_quantized = quantization(W_31, TRESHOLD)

      plot_connections(axs[j], N_quantized, W_31_quantized, positions_dict)

    os.makedirs(f'{PATH_TO_SHORTCUT_SAVE_FOLDER}architectures', exist_ok=True)

    plt.savefig(f'{PATH_TO_SHORTCUT_SAVE_FOLDER}architectures/architectures_beta{beta}_sample{sample}.png')