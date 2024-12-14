import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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
def plot_differences(data, inputs, targets, indices, model, device, title, file_name, best_test_rmse):
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
    plt.title(f"{title}  RMSE: {best_test_rmse:.3f} MeV")
    plt.savefig(file_name)
    plt.close()
    return


def plot_differences_nuclear_masses(data, inputs, targets, indices, model, device, title, file_name, best_test_rmse):
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
    plt.title(f"{title}  RMSE: {best_test_rmse:.3f} MeV")
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


def plot_differences_new(data, real_values, predictions, title, file_name):
    diff = real_values - predictions
    scatter_data = pd.DataFrame({
        'N': data['N'],
        'Z': data['Z'],
        'diff': diff
    })
    plt.figure(figsize=(10, 6))
    vmin = scatter_data['diff'].min()
    vmax = scatter_data['diff'].max()
    vcenter = 0 if vmin < 0 and vmax > 0 else (vmin + vmax) / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    scatter = plt.scatter(scatter_data['N'], scatter_data['Z'], c=scatter_data['diff']*(-1),
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
    rmse = np.sqrt(np.mean(diff**2))
    plt.title(f"{title}  RMSE: {rmse:.3f} MeV")
    plt.savefig(file_name)
    plt.close()
    return
