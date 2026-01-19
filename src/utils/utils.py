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

images_format = config['training']['images_format']

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


def create_5x5_neighborhood(data, idx, data_feature, extra_features=None):
    if extra_features is None:
        extra_features = []

    current_n = data.iloc[idx]['N']
    current_z = data.iloc[idx]['Z']

    z_grid = np.zeros((5, 5))
    n_grid = np.zeros((5, 5))
    data_feature_grid = np.zeros((5, 5))
    data_feature_list = []

    extra_grids = {feature: np.zeros((5, 5)) for feature in extra_features}

    for i in range(-2, 3):
        for j in range(-2, 3):
            neighbor_n = current_n + i
            neighbor_z = current_z + j

            neighbor_idx = data[(data['N'] == neighbor_n) &
                                (data['Z'] == neighbor_z)].index

            ii, jj = i + 2, j + 2
            z_grid[ii, jj] = neighbor_z
            n_grid[ii, jj] = neighbor_n

            if len(neighbor_idx) > 0:
                row = data.iloc[neighbor_idx[0]]

                value = row[data_feature]
                data_feature_grid[ii, jj] = value
                data_feature_list.append(value)

                for feature in extra_features:
                    extra_grids[feature][ii, jj] = row[feature]
            else:
                data_feature_grid[ii, jj] = np.nan

    neighborhood_mean = np.mean(data_feature_list) if data_feature_list else 0
    data_feature_grid[np.isnan(data_feature_grid)] = neighborhood_mean
    data_feature_grid[2, 2] = 0

    if extra_features:
        return z_grid, n_grid, *extra_grids.values(), data_feature_grid
    else:
        return z_grid, n_grid, data_feature_grid


color_limits_storage = {}
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


def evaluate_single_nucleus(data, model, n_value, z_value, data_feature,
                           neighborhood_function, extra_features=None, device=None):
    if device is None:
        device = next(model.parameters()).device
    if extra_features is None:
        extra_features = []

    nucleus_idx = data.index[(data["N"] == n_value) & (data["Z"] == z_value)]
    if len(nucleus_idx) == 0:
        raise ValueError(f"Nucleus with N={n_value} and Z={z_value} not found.")
    nucleus_idx = nucleus_idx[0]

    z_grid, n_grid, *rest = neighborhood_function(
        data, nucleus_idx, data_feature, extra_features=extra_features
    )

    # (C, 5, 5)
    input_array = np.stack([z_grid, n_grid] + rest).astype(np.float32)

    # (1, C, 5, 5)
    input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(device)

    real_value = data.iloc[nucleus_idx][data_feature]
    model.eval()
    with torch.no_grad():
        predicted_value = model(input_tensor).item()

    return real_value, predicted_value, real_value - predicted_value


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
        plt.savefig(os.path.join(lr_folder, f'CNN-{model_name}_evolution_lr_{lr}.{images_format}'))
        plt.close()
        return


def plot_data(df, df_column, colorbar_label, filename, folder, cmap, vmin=None, vcenter=None, vmax=None, title_name=None):

    os.makedirs(folder, exist_ok=True)
    
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

    if filename == f'nuclear_mass_expteo_dif.{images_format}':
        cbar.set_ticks([vmin, vmin/2, vcenter, vmax/2, vmax])
        cbar.set_ticklabels([f"{vmin:.0f}", f"{vmin/2:.0f}", f"{vcenter:.0f}", f"{vmax/2:.0f}", f"{vmax:.0f}"])

    if filename == f'bind_exp_per_nucleon.{images_format}':
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
