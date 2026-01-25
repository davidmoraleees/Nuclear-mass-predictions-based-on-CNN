import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


color_limits_storage = {}
def plot_differences_nuclear_masses(data, cfg, inputs, targets, indices, model, device, file_name):
    uma = cfg['LDM']['uma']
    m_e = cfg['LDM']['m_e']
    m_n =  cfg['LDM']['m_n']*(10**-6)*uma 
    m_H =  cfg['LDM']['m_H']*(10**-6)*uma 

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


def plot_evolution(train_loss_rmse_values, test_loss_rmse_values, plot_skipping_epochs, lr, output_path, model_name):
        plt.figure(figsize=(10, 5))
        epochs_used = len(train_loss_rmse_values)
        plt.plot(range(plot_skipping_epochs, epochs_used + 1), train_loss_rmse_values[plot_skipping_epochs-1:],
                 label=r'Training $\sigma$ (MeV)', color='blue', linewidth=0.5)
        plt.plot(range(plot_skipping_epochs, epochs_used + 1), test_loss_rmse_values[plot_skipping_epochs-1:],
                 label=r'Testing $\sigma$ (MeV)', color='red', linewidth=0.5)
        plt.xlabel('Epoch')
        plt.ylabel(r'$\sigma$ (MeV)')
        max_value = max(max(train_loss_rmse_values[plot_skipping_epochs-1:]),
                        max(test_loss_rmse_values[plot_skipping_epochs-1:])) + 1
        plt.xlim(plot_skipping_epochs, epochs_used + 1)
        plt.ylim(0, max_value)
        plt.title(r'Evolution of $\sigma$ for ' + f'{model_name}') 
        plt.legend()
        plt.tick_params(axis='x', pad=10)
        plt.tick_params(axis='y', pad=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return


def plot_data(df, df_column, colorbar_label, output_path, cmap,
              vmin=None, vcenter=None, vmax=None, title_name=None):
    
    plt.figure(figsize=(10, 8))

    if vmin is None:
        vmin = df[df_column].min()
    if vmax is None:
        vmax = df[df_column].max()
    if vcenter is None:
        vcenter = 0 if vmin < 0 and vmax > 0 else (vmin + vmax) / 2

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    scatter = plt.scatter(df['N'], df['Z'], c=df[df_column], cmap=cmap, norm=norm, edgecolor='None', s=14)
    cbar = plt.colorbar(scatter, orientation="horizontal", fraction=0.08, shrink=0.5)
    cbar.set_label(colorbar_label)

    magic_numbers = [8, 20, 28, 50, 82, 126]
    for m in magic_numbers:
        plt.axvline(m, color="gray", linestyle="--", linewidth=0.5)
        plt.axhline(m, color="gray", linestyle="--", linewidth=0.5)

    plt.xticks(magic_numbers)
    plt.yticks(magic_numbers)
    plt.xlabel("N")
    plt.ylabel("Z")
    plt.title(title_name)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
