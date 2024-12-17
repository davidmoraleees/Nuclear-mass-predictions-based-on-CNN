import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import yaml
from models import CNN_I3
from utils import create_5x5_neighborhood_i3
from utils import plot_differences, plot_differences_nuclear_masses
from utils import train_model, plot_evolution
from utils import fontsizes

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on:', device)

csv_file = config['data']['csv_file'] 
data = pd.read_csv(csv_file, delimiter=';')
data_feature = config['data']['data_feature']
num_epochs = config['training']['num_epochs']
patience = config['training']['patience']
learning_rate = config['training']['learning_rate']
optimizer_name = config['training']['optimizer_name']
plot_skipping_epochs = config['training']['plot_skipping_epochs']
I3_results_folder = 'CNN-I3 results'
I3_lr_folder = 'CNN-I3 experiments learning rates'
model_name = 'I3'

fontsizes(config)

inputs = [] #3x5x5 matrices of inputs
targets = [] #Binding energies of the target nucleus

for idx in range(len(data)):
    z_grid, n_grid, data_feature_grid = create_5x5_neighborhood_i3(data, idx, data_feature)
    input_grid = np.stack([z_grid, n_grid, data_feature_grid], axis=0)
    inputs.append(input_grid)
    target_value = data.iloc[idx][data_feature]
    targets.append(target_value)

inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(device) #shape: (n, 3, 5, 5) where 'n' is the number of nucleus
targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1).to(device) #shape: (n, 1)

indices = np.arange(len(data)) #Original indices in the Dataframe
train_inputs, test_inputs, train_targets, test_targets, train_indices, test_indices = train_test_split(
    inputs_tensor,targets_tensor, indices, test_size=config['data']['test_size'], shuffle=True,
    random_state=config['general']['random_state'])

# Learning rates study
learning_rates = [0.00001, 0.00005]

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    
    model = CNN_I3().to(device)
    train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch_test = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, lr, optimizer_name, patience, I3_lr_folder, model_name, lr)

    plot_evolution(train_loss_rmse_values, test_loss_rmse_values, plot_skipping_epochs, num_epochs, lr, I3_lr_folder, model_name)
    
    color_limits_storage = {}
    color_limits_storage['color_limits'] = (-6, 0, 6)
    plot_differences(data, inputs_tensor, targets_tensor, range(len(data)), model, device,
                     f'Difference exp-predicted (all data, lr={lr})',
                     f"{I3_lr_folder}/CNN-I3_diff_scatter_lr_{lr}.png", best_test_rmse)
    
    plot_differences(data, train_inputs, train_targets, train_indices, model, device,
                     f'Difference exp-predicted (training set, lr={lr})',
                     f"{I3_lr_folder}/CNN-I3_diff_scatter_train_lr_{lr}.png", best_test_rmse)
    
    plot_differences(data, test_inputs, test_targets, test_indices, model, device,
                     f'Difference exp-predicted (test set, lr={lr})',
                     f"{I3_lr_folder}/CNN-I3_diff_scatter_test_lr_{lr}.png", best_test_rmse)
    
    # Now we convert total binding energy predictions into nuclear mass predictions
    color_limits_storage['color_limits'] = (-6, 0, 6)
    plot_differences_nuclear_masses(data, inputs_tensor, targets_tensor, range(len(data)), model, device,
                                    'Difference exp-predicted (all data) nuclear masses', f'{I3_lr_folder}/CNN-I3_diff_scatter_nuclear_masses_lr_{lr}.png', best_test_rmse)

    plot_differences_nuclear_masses(data, train_inputs, train_targets, train_indices, model, device,
                                    'Difference exp-predicted (training set) nuclear masses', f'{I3_lr_folder}/CNN-I3_diff_scatter_train_nuclear_masses_lr_{lr}.png', best_test_rmse)

    plot_differences_nuclear_masses(data, test_inputs, test_targets, test_indices, model, device,
                                    'Difference exp-predicted (test set) nuclear masses', f'{I3_lr_folder}/CNN-I3_diff_scatter_test_nuclear_masses_lr_{lr}.png', best_test_rmse)
