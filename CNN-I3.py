import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import yaml
from models import CNN_I3
from utils import create_5x5_neighborhood_i3
from utils import plot_differences, plot_differences_nuclear_masses

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


def save_model(model, folder, best_model_state, best_test_rmse, best_epoch, num_epochs, lr_name=None):
    lr_value = lr_name if lr_name is not None else ''

    if best_model_state is not None:
            torch.save(best_model_state, f'{folder}/cnn_i3_best_model_{lr_value}.pt')
            print(f'Best RMSE: {best_test_rmse:.4f}MeV found in epoch {best_epoch}')
    else:
        torch.save(model.state_dict(), f'{folder}/cnn_i3_model_{num_epochs}_epochs_{lr_value}.pt')
        print('Best model not found. Saving last model')
    return


def load_model(model, folder, best_model_state, best_test_rmse, best_epoch, num_epochs):
    if best_model_state is not None:
        model.load_state_dict(best_model_state, map_location=device)
        print(f'Model loaded from epoch {best_epoch} with RMSE: {best_test_rmse:.4f} MeV')
    else:
        model.load_state_dict(torch.load(f'{folder}/cnn_i3_model_{num_epochs}_epochs.pt', map_location=device))
        print('Best model not found. Loading last model')
    return


def train_model(model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate, optimizer_name, patience, folder, lr_name=None):
    criterion = nn.MSELoss() #Instance of the MSE
    OptimizerClass = getattr(optim, optimizer_name)
    optimizer = OptimizerClass(model.parameters(), lr=learning_rate) #model.parameters()=weights and biases to optimize
    #lr=how much to adjust the model's parameters with respect to the loss gradient in each epoch.
    #Adam=adaptative moment estimation. It calculates a separate learning rate for each parameter

    print(f'Total number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    #p.numel() counts the number of elements that has every tensor. 
    #We only count those which are used for training (requires_grad=True).

    train_loss_rmse_values = []
    test_loss_rmse_values = []
    best_test_rmse = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad() #Reset of gradients to zero to avoid accumulation from previous runs
        train_outputs = model(train_inputs.to(device))
        train_loss = criterion(train_outputs, train_targets)
        train_loss.backward() #Gradients of the loss with respect to model parameters using backpropagation
        optimizer.step() #We update the model parameters using the calculated gradients
        train_loss_rmse = torch.sqrt(train_loss)
        train_loss_rmse_values.append(train_loss_rmse.item())

        model.eval()
        with torch.no_grad(): #We disable gradient calculation for the test phase
            test_outputs = model(test_inputs.to(device))
            test_loss_mse = criterion(test_outputs, test_targets)
            test_loss_rmse = torch.sqrt(test_loss_mse)
            test_loss_rmse_values.append(test_loss_rmse.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train loss: {train_loss_rmse.item():.4f} MeV, Test loss: {test_loss_rmse.item():.4f} MeV')
                  
        if test_loss_rmse.item() < best_test_rmse:
            best_test_rmse = test_loss_rmse.item()
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'There was no improvement in {patience} epochs. Training stopped.')
            break

    save_model(model, folder, best_model_state, best_test_rmse, best_epoch, num_epochs, lr_name)
    return train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch


# Learning rates study
learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5]

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    
    model = CNN_I3().to(device)
    train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, lr, optimizer_name, patience, I3_lr_folder, lr)
    
    plt.figure(figsize=(10, 5))
    epochs_used = len(train_loss_rmse_values)
    plt.plot(range(plot_skipping_epochs, epochs_used + 1), train_loss_rmse_values[plot_skipping_epochs-1:], label='Training RMSE', color='blue', linewidth=0.5)
    plt.plot(range(plot_skipping_epochs, epochs_used + 1), test_loss_rmse_values[plot_skipping_epochs-1:], label='Test RMSE', color='red', linewidth=0.5)
    plt.title(f'Evolution of RMSE over {num_epochs} epochs (lr={lr})')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (MeV)')
    max_value = max(max(train_loss_rmse_values[plot_skipping_epochs-1:]), max(test_loss_rmse_values[plot_skipping_epochs-1:])) + 1
    plt.xlim(plot_skipping_epochs, epochs_used + 1)
    plt.ylim(0, max_value) 
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(I3_lr_folder, f'CNN-I3_evolution_lr_{lr}.png'))
    plt.close()
    
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

 
# One training of the model
model = CNN_I3().to(device) #Instance of our model
train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch = train_model(
    model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate, optimizer_name, patience, I3_results_folder)

plt.figure(figsize=(10, 5))
epochs_used = len(train_loss_rmse_values)
plt.plot(range(plot_skipping_epochs, epochs_used + 1), train_loss_rmse_values[plot_skipping_epochs-1:], label='Training RMSE', color='blue', linewidth=0.5)
plt.plot(range(plot_skipping_epochs, epochs_used + 1), test_loss_rmse_values[plot_skipping_epochs-1:], label='Test RMSE', color='red', linewidth=0.5)
plt.title(f'Evolution of RMSE over {num_epochs} epochs')
plt.xlabel('Època')
plt.ylabel('RMSE (MeV)')
max_value = max(max(train_loss_rmse_values[plot_skipping_epochs-1:]), max(test_loss_rmse_values[plot_skipping_epochs-1:])) + 1
plt.xlim(plot_skipping_epochs, epochs_used + 1)
plt.ylim(0, max_value) 
plt.legend()
plt.grid()
plt.savefig(f'{I3_results_folder}/CNN-I3_evolution.png')
plt.close()

color_limits_storage = {}
plot_differences(data, inputs_tensor, targets_tensor, range(len(data)), model, device,
                'Difference exp-predicted (all data)', f'{I3_results_folder}/CNN-I3_diff_scatter.png', best_test_rmse)

plot_differences(data, train_inputs, train_targets, train_indices, model, device,
                'Difference exp-predicted (training set)', f'{I3_results_folder}/CNN-I3_diff_scatter_train.png', best_test_rmse)

plot_differences(data, test_inputs, test_targets, test_indices, model, device,
                'Difference exp-predicted (test set)', f'{I3_results_folder}/CNN-I3_diff_scatter_test.png', best_test_rmse)

# Now we convert total binding energy predictions into nuclear mass predictions
color_limits_storage = {}
plot_differences_nuclear_masses(data, inputs_tensor, targets_tensor, range(len(data)), model, device,
                                'Difference exp-predicted (all data) nuclear masses', f'{I3_results_folder}/CNN-I3_diff_scatter_nuclear_masses.png', best_test_rmse)

plot_differences_nuclear_masses(data, train_inputs, train_targets, train_indices, model, device,
                                'Difference exp-predicted (training set) nuclear masses', f'{I3_results_folder}/CNN-I3_diff_scatter_train_nuclear_masses.png', best_test_rmse)

plot_differences_nuclear_masses(data, test_inputs, test_targets, test_indices, model, device,
                                'Difference exp-predicted (test set) nuclear masses', f'{I3_results_folder}/CNN-I3_diff_scatter_test_nuclear_masses.png', best_test_rmse)


# K-folding
n_splits = config['kfolding']['n_splits']
kf = KFold(n_splits=n_splits, shuffle=True, random_state=config['general']['random_state'])

rmse_train_list = []
rmse_test_list = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs_tensor)):
    print(f"Fold {fold + 1}/{n_splits}")

    train_inputs, test_inputs = inputs_tensor[train_idx], inputs_tensor[test_idx]
    train_targets, test_targets = targets_tensor[train_idx], targets_tensor[test_idx]
    model = CNN_I3().to(device)

    train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate, optimizer_name, patience, I3_results_folder)
    
    rmse_train_list.append(min(train_loss_rmse_values))
    rmse_test_list.append(best_test_rmse) 

    print(f"Fold {fold + 1}: Best RMSE (Test): {best_test_rmse:.4f}MeV in epoch {best_epoch}")

mean_rmse_train = np.mean(rmse_train_list)
std_rmse_train = np.std(rmse_train_list)
mean_rmse_test = np.mean(rmse_test_list)
std_rmse_test = np.std(rmse_test_list)

print(f"Final Average Train RMSE: {mean_rmse_train:.4f} ± {std_rmse_train:.4f} MeV")
print(f"Final Average Test RMSE: {mean_rmse_test:.4f} ± {std_rmse_test:.4f} MeV")
