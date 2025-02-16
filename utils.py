import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import yaml
import datetime
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def fontsizes(config):
    plt.rcParams.update({
        'font.size': config['fontsizes']['font_size'],
        'axes.titlesize': config['fontsizes']['axes_title_size'],
        'axes.labelsize': config['fontsizes']['axes_label_size'],
        'xtick.labelsize': config['fontsizes']['xtick_labelsize'],
        'ytick.labelsize': config['fontsizes']['ytick_labelsize'],
        'legend.fontsize': config['fontsizes']['legend_fontsize'],
        'figure.titlesize': config['fontsizes']['figure_title_size'],
        'font.family': 'serif',  # Fuente serif
        'font.serif': ['STIX', 'Times New Roman', 'DejaVu Serif'],  # Fuentes de reserva
        'mathtext.fontset': 'stix',
    })
    return

fontsizes(config)


def create_5x5_neighborhood_i3(data, idx, data_feature):
    current_n = data.iloc[idx]['N']
    current_z = data.iloc[idx]['Z']
    
    z_grid = np.zeros((5, 5))
    n_grid = np.zeros((5, 5))
    data_feature_grid = np.zeros((5, 5))
    data_feature_list = []
    
    for i in range(-2, 3):
        for j in range(-2, 3):
            neighbor_n = current_n + i
            neighbor_z = current_z + j
            neighbor_idx = data[(data['N'] == neighbor_n) & (data['Z'] == neighbor_z)].index
            
            z_grid[i+2, j+2] = neighbor_z
            n_grid[i+2, j+2] = neighbor_n  

            if len(neighbor_idx) > 0:
                data_feature_value = data.iloc[neighbor_idx[0]][data_feature]
                data_feature_grid[i + 2, j + 2] = data_feature_value
                data_feature_list.append(data_feature_value)
            else:
                data_feature_grid[i + 2, j + 2] = np.nan 

    if data_feature_list:
        neighborhood_mean = np.mean(data_feature_list)
    else:
        neighborhood_mean = 0

    data_feature_grid[np.isnan(data_feature_grid)] = neighborhood_mean
    data_feature_grid[2, 2] = 0
    return z_grid, n_grid, data_feature_grid


def create_5x5_neighborhood_i4(data, idx, data_feature):
    current_n = data.iloc[idx]['N'] #Data for the target nucleus. 'idx'=row and 'N'=column
    current_z = data.iloc[idx]['Z']
    
    z_grid = np.zeros((5, 5))
    n_grid = np.zeros((5, 5))
    delta_I4_grid = np.zeros((5, 5))
    data_feature_grid = np.zeros((5, 5))
    data_feature_list = []
    
    for i in range(-2, 3): #The neighbourhood is defined from -2 to 2, 0 being the central value
        for j in range(-2, 3):
            neighbor_n = current_n + i #Data of the neighbours of the target nucleus.
            neighbor_z = current_z + j
            neighbor_idx = data[(data['N'] == neighbor_n) & (data['Z'] == neighbor_z)].index #row index of the neighbour
                                                                                             #that has 'neighbor_n' and 'neighbor_z' 
            z_grid[i+2, j+2] = neighbor_z #We add +2 because matrices start at [0,0] (top left corner)
            n_grid[i+2, j+2] = neighbor_n  

            if len(neighbor_idx) > 0: #Verify if any index has been found
                data_feature_value = data.iloc[neighbor_idx[0]][data_feature]
                data_feature_grid[i + 2, j + 2] = data_feature_value
                data_feature_list.append(data_feature_value)
                delta_I4_value = data.iloc[neighbor_idx[0]]['delta_I4']
                delta_I4_grid[i + 2, j + 2] = delta_I4_value  
            else:
                data_feature_grid[i + 2, j + 2] = np.nan 

    if data_feature_list:
        neighborhood_mean = np.mean(data_feature_list)
    else:
        print('Warning: data feature list is empty. Proceeding with neighborhood_mean = 0')
        neighborhood_mean = 0

    data_feature_grid[np.isnan(data_feature_grid)] = neighborhood_mean
    data_feature_grid[2, 2] = 0 #Target nucleus assigned to zero
    return z_grid, n_grid, delta_I4_grid, data_feature_grid


color_limits_storage = {}
def plot_differences(data, inputs, targets, indices, model, device, file_name, best_test_rmse):
    model.eval() 
    with torch.no_grad():
        outputs = model(inputs.to(device)).cpu().numpy()
        targets_np = targets.cpu().numpy()
        diff = targets_np - outputs

    scatter_data = pd.DataFrame({
        'N': data.iloc[indices]['N'].values,
        'Z': data.iloc[indices]['Z'].values,
        'diff': diff.flatten()})

    plt.figure(figsize=(10, 6))

    if 'color_limits' not in color_limits_storage:
        vmin = scatter_data['diff'].min()
        vmax = scatter_data['diff'].max()
        vcenter = 0 if vmin < 0 and vmax > 0 else (vmin + vmax) / 2
        color_limits_storage['color_limits'] = (vmin, vcenter, vmax)
    else:
        vmin, vcenter, vmax = color_limits_storage['color_limits']

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    scatter = plt.scatter(scatter_data['N'], scatter_data['Z'], c=scatter_data['diff'],
                          cmap='seismic', norm=norm, edgecolor='None', s=12)
    cbar = plt.colorbar(scatter)
    cbar.set_label('(MeV)')
    magic_numbers = [8, 20, 28, 50, 82, 126]
    for magic in magic_numbers:
        plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(magic_numbers)
    plt.yticks(magic_numbers)
    plt.xlabel('N')
    plt.ylabel('Z')
    plt.savefig(file_name)
    plt.close()
    return


def plot_differences_nuclear_masses(data, inputs, targets, indices, model, device, file_name, best_test_rmse):
    uma = config['LDM']['uma']
    m_e = config['LDM']['m_e']
    m_n =  config['LDM']['m_n']*(10**-6)*uma 
    m_H =  config['LDM']['m_H']*(10**-6)*uma 

    model.eval() 
    with torch.no_grad():
        outputs = model(inputs.to(device)).cpu().numpy().flatten()
        outputs = data['Z'].iloc[indices]*m_H + data['N'].iloc[indices]*m_n - outputs # Calculating atomic masses
        outputs = outputs - data['Z'].iloc[indices]*m_e + data['B_e'].iloc[indices] # Calculating nuclear masses

        targets_np = targets.cpu().numpy().flatten()
        targets_np = data['Z'].iloc[indices]*m_H + data['N'].iloc[indices]*m_n - targets_np # Calculating atomic masses
        targets_np = targets_np - data['Z'].iloc[indices]*m_e + data['B_e'].iloc[indices] # Calculating nuclear masses

        diff = targets_np - outputs

    scatter_data = pd.DataFrame({
        'N': data.iloc[indices]['N'].values,
        'Z': data.iloc[indices]['Z'].values,
        'diff': diff.to_numpy()})

    plt.figure(figsize=(10, 6))

    if 'color_limits' not in color_limits_storage:
        vmin = scatter_data['diff'].min()
        vmax = scatter_data['diff'].max()
        vcenter = 0 if vmin < 0 and vmax > 0 else (vmin + vmax) / 2
        color_limits_storage['color_limits'] = (vmin, vcenter, vmax)
    else:
        vmin, vcenter, vmax = color_limits_storage['color_limits']

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    scatter = plt.scatter(scatter_data['N'], scatter_data['Z'], c=scatter_data['diff'],
                          cmap='seismic', norm=norm, edgecolor='None', s=12)
    cbar = plt.colorbar(scatter)
    cbar.set_label('(MeV)')
    magic_numbers = [8, 20, 28, 50, 82, 126]
    for magic in magic_numbers:
        plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(magic_numbers)
    plt.yticks(magic_numbers)
    plt.xlabel('N')
    plt.ylabel('Z')
    plt.savefig(file_name)
    plt.close()
    return


def evaluate_single_nucleus(data, model, n_value, z_value, data_feature, neighborhood_function):
    nucleus_idx = data[(data['N'] == n_value) & (data['Z'] == z_value)].index
    if len(nucleus_idx) == 0:
        raise ValueError(f"Nucleus with N={n_value} and Z={z_value} not found.")
    
    nucleus_idx = nucleus_idx[0]  # If there's more than one, we take the first.
    neighborhood = neighborhood_function(data, nucleus_idx, data_feature)
    
    if neighborhood_function == create_5x5_neighborhood_i3:
        z_grid, n_grid, data_feature_grid = neighborhood
        input_tensor = torch.tensor(np.array([np.stack([z_grid, n_grid, data_feature_grid])]), dtype=torch.float32).to(device)
    elif neighborhood_function == create_5x5_neighborhood_i4:
        z_grid, n_grid, delta_I4_grid, data_feature_grid = neighborhood
        input_tensor = torch.tensor(np.array([np.stack([z_grid, n_grid, delta_I4_grid, data_feature_grid])]), dtype=torch.float32).to(device)
    else:
        raise ValueError("Error: neighborhood_function not recognized.")
    
    real_value = data.iloc[nucleus_idx][data_feature]
    model.eval()
    with torch.no_grad():
        predicted_value = model(input_tensor).item()
    difference = real_value - predicted_value
    
    return real_value, predicted_value, difference


def plot_differences_new(data, real_values, predictions, file_name, title_name=None):
    diff = real_values - predictions
    scatter_data = pd.DataFrame({
        'N': data['N'],
        'Z': data['Z'],
        'diff': diff
    })
    plt.figure(figsize=(10, 8))

    if 'color_limits' not in color_limits_storage:
        vmin = scatter_data['diff'].min()
        vmax = scatter_data['diff'].max()
        vcenter = 0 if vmin < 0 and vmax > 0 else (vmin + vmax) / 2
        color_limits_storage['color_limits'] = (vmin, vcenter, vmax)
    else:
        vmin, vcenter, vmax = color_limits_storage['color_limits']

    vmin = -3
    vcenter = 0
    vmax = 3

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    scatter = plt.scatter(scatter_data['N'], scatter_data['Z'], c=scatter_data['diff']*(-1),
                          cmap='seismic', norm=norm, edgecolor='None', s=12)
    cbar = plt.colorbar(scatter, orientation='horizontal', fraction=0.08, shrink=0.5)
    cbar.set_ticks([vmin, vmin/2, vcenter, vmax/2, vmax])
    cbar.set_ticklabels([f"{vmin:.0f}", f"{vmin/2:.0f}", f"{vcenter:.0f}", f"{vmax/2:.0f}", f"{vmax:.0f}"])
    cbar.set_label(r'$\Delta$ (MeV)')

    magic_numbers = [8, 20, 28, 50, 82, 126]
    for magic in magic_numbers:
        plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(magic_numbers)
    plt.yticks(magic_numbers)

    xtick_positions = magic_numbers
    xtick_labels = [str(magic) for magic in magic_numbers]
    xtick_labels[1] = "20 "
    xtick_labels[2] = "  28"
    plt.gca().set_xticks(xtick_positions)
    plt.gca().set_xticklabels(xtick_labels)

    plt.xlabel('N')
    plt.ylabel('Z')
    plt.title(title_name)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    return


def plot_differences_combined(data_i3, diff_i3, data_i4, diff_i4, data_ldm, diff_ldm, file_name):

    fig, axes = plt.subplots(2, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    #vmin_ldm = -14
    #vmax_ldm = 14
    #vcenter_ldm = 0 if vmin_ldm < 0 and vmax_ldm > 0 else (vmin_ldm + vmax_ldm) / 2
    #norm_ldm = TwoSlopeNorm(vmin=vmin_ldm, vcenter=vcenter_ldm, vmax=vmax_ldm)

    vmin_cnn = -3
    vmax_cnn = 3
    vcenter_cnn = 0 if vmin_cnn < 0 and vmax_cnn > 0 else (vmin_cnn + vmax_cnn) / 2
    norm_cnn = TwoSlopeNorm(vmin=vmin_cnn, vcenter=vcenter_cnn, vmax=vmax_cnn)

    #scatter1 = axes[0].scatter(data_ldm['N'], data_ldm['Z'], c=diff_ldm*(-1),
    #                            cmap='seismic', norm=norm_ldm, edgecolor='None', s=12)
    #axes[0].set_title("LDM")
    #axes[0].set_xlabel("N")
    #axes[0].set_ylabel("Z")

    scatter1 = axes[0].scatter(data_i3['N'], data_i3['Z'], c=diff_i3*(-1),
                                cmap='seismic', norm=norm_cnn, edgecolor='None', s=12)
    axes[0].set_title("I3")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Z")

    scatter2 = axes[1].scatter(data_i4['N'], data_i4['Z'], c=diff_i4*(-1),
                                cmap='seismic', norm=norm_cnn, edgecolor='None', s=12)
    axes[1].set_title("I4")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Z")

    magic_numbers = [8, 20, 28, 50, 82, 126]
    for ax in axes:
        for magic in magic_numbers:
            ax.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
            ax.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
        ax.set_yticks(magic_numbers)

        xtick_positions = magic_numbers
        xtick_labels = [str(magic) for magic in magic_numbers]
        xtick_labels[1] = "20 "
        xtick_labels[2] = "  28"
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)

        ax.grid(alpha=0.3)

    #cbar1 = fig.colorbar(scatter1, ax=axes[0], orientation='vertical', fraction=0.08)
    #cbar1.set_label("(MeV)")

    cbar1 = fig.colorbar(scatter1, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.03, shrink=0.5, pad=0.07)
    cbar1.set_label(r'$\Delta$ (MeV)')
    cbar1.set_ticks([vmin_cnn, vmin_cnn/2, vcenter_cnn, vmax_cnn/2, vmax_cnn])
    cbar1.set_ticklabels([f"{vmin_cnn:.0f}", f"{vmin_cnn/2}", f"{vcenter_cnn:.0f}", f"{vmax_cnn/2}", f"{vmax_cnn:.0f}"])
    
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    return


def save_model(model, folder, best_model_state, best_test_rmse, best_epoch, num_epochs, model_name, lr_name=None):
    lr_value = lr_name if lr_name is not None else ''

    if best_model_state is not None:
            torch.save(best_model_state, f'{folder}/cnn_{model_name}_best_model_{lr_value}.pt')
            print(f'Best RMSE: {best_test_rmse:.4f}MeV found in epoch {best_epoch}')
    else:
        torch.save(model.state_dict(), f'{folder}/cnn_{model_name}_model_{num_epochs}_epochs_{lr_value}.pt')
        print('Best model not found. Saving last model')
    return


def load_model(model, folder, best_model_state, best_test_rmse, best_epoch, num_epochs, model_name):
    if best_model_state is not None:
        model.load_state_dict(best_model_state, map_location=device)
        print(f'Model loaded from epoch {best_epoch} with RMSE: {best_test_rmse:.4f} MeV')
    else:
        model.load_state_dict(torch.load(f'{folder}/cnn_{model_name}_model_{num_epochs}_epochs.pt', map_location=device))
        print('Best model not found. Loading last model')
    return


def train_model(model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate, optimizer_name, patience, folder, model_name, lr_name=None):
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
    best_train_rmse = float('inf')
    best_test_rmse = float('inf')
    best_model_state = None
    best_epoch_train = 0
    best_epoch_test = 0
    epochs_without_improvement = 0

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

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

        if train_loss_rmse.item() < best_train_rmse:
            best_train_rmse = train_loss_rmse.item()
            best_epoch_train = epoch + 1

        if test_loss_rmse.item() < best_test_rmse:
            best_test_rmse = test_loss_rmse.item()
            best_model_state = model.state_dict()
            best_epoch_test = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'There was no improvement in {patience} epochs. Training stopped.')
            break

    save_model(model, folder, best_model_state, best_test_rmse, best_epoch_test, num_epochs, model_name, lr_name)

    end_time = datetime.datetime.now()
    end_time_str = end_time.strftime('%H:%M:%S %d/%m/%Y')

    logs_filename = f"{folder}/training_logs_{model_name}.txt"
    
    with open(logs_filename, 'a') as f:
        f.write(f"Execution started at: {start_time_str}\n")
        f.write(f"Model trained: CNN-{model_name}\n")
        f.write(f"Optimizer: {optimizer_name}, Learning rate: {learning_rate}\n")
        f.write(f"Predefined number of epochs: {num_epochs}, Patience: {patience}\n")
        f.write(f"Best train RMSE: {best_train_rmse:.4f} MeV, Best train epoch: {best_epoch_train}\n")
        f.write(f"Best test RMSE: {best_test_rmse:.4f} MeV, Best test epoch: {best_epoch_test}\n")
        f.write(f"Execution ended at: {end_time_str}\n\n\n")

    return train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch_test


def plot_evolution(train_loss_rmse_values, test_loss_rmse_values, plot_skipping_epochs, num_epochs, lr, lr_folder, model_name):
        plt.figure(figsize=(10, 5))
        epochs_used = len(train_loss_rmse_values)
        plt.plot(range(plot_skipping_epochs, epochs_used + 1), train_loss_rmse_values[plot_skipping_epochs-1:], label=r'Training $\sigma$ (MeV)', color='blue', linewidth=0.5)
        plt.plot(range(plot_skipping_epochs, epochs_used + 1), test_loss_rmse_values[plot_skipping_epochs-1:], label=r'Testing $\sigma$ (MeV)', color='red', linewidth=0.5)
        plt.xlabel('Epoch')
        plt.ylabel(r'$\sigma$ (MeV)')
        max_value = max(max(train_loss_rmse_values[plot_skipping_epochs-1:]), max(test_loss_rmse_values[plot_skipping_epochs-1:])) + 1
        plt.xlim(plot_skipping_epochs, epochs_used + 1)
        plt.ylim(0, max_value)
        plt.title(r'Evolution of $\sigma$ for ' + f'{model_name}') 
        plt.legend()
        plt.tick_params(axis='x', pad=10)
        plt.tick_params(axis='y', pad=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(lr_folder, f'CNN-{model_name}_evolution_lr_{lr}.pdf'))
        plt.close()
        return


def plot_data(df, df_column, colorbar_label, filename, folder, cmap, vmin=None, vcenter=None, vmax=None, title_name=None):
    plt.figure(figsize=(10, 8))

    if vmin is None:
        vmin = df[df_column].min()
    if vmax is None:
        vmax = df[df_column].max()
    if vcenter is None:
        vcenter = 0 if vmin < 0 and vmax > 0 else (vmin + vmax) / 2

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    scatter = plt.scatter(df['N'], df['Z'], c=df[df_column], cmap=cmap, norm=norm, edgecolor='None', s=14)
    cbar = plt.colorbar(scatter, orientation='horizontal', fraction=0.08, shrink=0.5)
    cbar.set_label(colorbar_label)

    if filename == 'nuclear_mass_expteo_dif.pdf':
        cbar.set_ticks([vmin, vmin/2, vcenter, vmax/2, vmax])
        cbar.set_ticklabels([f"{vmin:.0f}", f"{vmin/2:.0f}", f"{vcenter:.0f}", f"{vmax/2:.0f}", f"{vmax:.0f}"])

    if filename == 'bind_exp_per_nucleon.pdf':
        cbar.set_ticks([vmin, vcenter, vmax])
        cbar.set_ticklabels([f"{vmin:.0f}", f"{vcenter:.0f}", f"{vmax:.0f}"])

    magic_numbers = [8, 20, 28, 50, 82, 126]
    for magic in magic_numbers:
        plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)

    plt.xticks(magic_numbers)
    plt.yticks(magic_numbers)

    xtick_positions = magic_numbers
    xtick_labels = [str(magic) for magic in magic_numbers]
    xtick_labels[1] = "20 "
    xtick_labels[2] = "  28"
    plt.gca().set_xticks(xtick_positions)
    plt.gca().set_xticklabels(xtick_labels)

    plt.xlabel('N')
    plt.ylabel('Z')
    plt.title(title_name) 
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close()
    return
