import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from models import CNN_I3, CNN_I4
from utils import create_5x5_neighborhood_i3, create_5x5_neighborhood_i4

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def plot_differences(data, real_values, predictions, title, file_name):
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
    rmse = np.sqrt(np.mean(diff**2))
    plt.title(f"{title}  RMSE: {rmse:.3f} MeV")
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
csv_file = "data/mass2020_cleaned_with_#.csv"
new_nuclei_file = "data/df2016_2020_yesyes.csv"  
model_path = "Tests new nuclei/cnn_i3_best_model_1e-05.pt"  
data_feature = config['data']['data_feature'] 

data = pd.read_csv(csv_file, delimiter=';')
new_nuclei = pd.read_csv(new_nuclei_file, delimiter=';')


# Evaluation of specific nuclei
z_value = 109
n_range = range(156, 174)  # Mt isotopic chain
nuclei_to_evaluate = [(n_value, z_value) for n_value in n_range]

models_config = [
    {"name": "CNN-I3", "model_path": "Tests new nuclei/cnn_i3_best_model_1e-05.pt", "model_class": CNN_I3, "neighborhood_func": create_5x5_neighborhood_i3},
    {"name": "CNN-I4", "model_path": "Tests new nuclei/cnn_i4_best_model_1e-05.pt", "model_class": CNN_I4, "neighborhood_func": create_5x5_neighborhood_i4}
]

av = config['LDM']['av']
aS = config['LDM']['aS']
ac = config['LDM']['ac']
aA = config['LDM']['aA']
ap = config['LDM']['ap']

results = []

for model_config in models_config:
    model = model_config["model_class"]().to(device)
    model.load_state_dict(torch.load(model_config["model_path"], map_location=device, weights_only=True))

    for n_value, z_value in nuclei_to_evaluate:
        try:
            real_value, predicted_value, difference = evaluate_single_nucleus(
                data, model, n_value, z_value, data_feature, model_config["neighborhood_func"]
            )
            results.append({"N": n_value, "Z": z_value, "Real Value (MeV)": real_value, 
                            "Predicted Value (MeV)": predicted_value, "Difference (MeV)": difference, 
                            "Model": model_config["name"]})
        except ValueError as e:
            print(f"Error for Model={model_config['name']}, N={n_value}, Z={z_value}: {e}")

for n_value, z_value in nuclei_to_evaluate:
    A = z_value + n_value
    delta = -1 if (z_value % 2 == 0 and n_value % 2 == 0) else 0 if A % 2 == 1 else (1 if z_value % 2 == 1 and n_value % 2 == 1 else np.nan)
    ldm_mass = (av * A - aS * A**(2/3) - ac * (z_value**2) * A**(-1/3) - aA * ((A - 2 * z_value)**2) / A - ap * delta * A**(-1/2))
    real_value = data.loc[(data['N'] == n_value) & (data['Z'] == z_value), data_feature].values
    if real_value.size > 0:
        real_value = real_value[0]
        ldm_difference = real_value - ldm_mass
        results.append({"N": n_value, "Z": z_value, "Real Value (MeV)": real_value, 
                        "Predicted Value (MeV)": ldm_mass, "Difference (MeV)": ldm_difference, 
                        "Model": "LDM"})

results_df = pd.DataFrame(results)
new_nuclei_set = set(zip(new_nuclei['N'], new_nuclei['Z']))
results_df['In 2020?'] = results_df.apply(lambda row: (row['N'], row['Z']) in new_nuclei_set, axis=1)
output_csv_file = "Tests new nuclei/predictions_nuclei.csv"
results_df.to_csv(output_csv_file, sep=";", index=False)

colors = {"CNN-I3": "blue", "CNN-I4": "red", "LDM": "green"}
plt.figure(figsize=(10, 6))
plt.axvspan(156, 170.5, color='gray', alpha=0.3)

for model_name in ["CNN-I3", "CNN-I4", "LDM"]:
    model_data = results_df[results_df["Model"] == model_name]
    plt.plot(model_data["N"], model_data["Difference (MeV)"], marker="o", label=model_name, color=colors[model_name])

plt.axhline(0, color='black', linewidth=2, linestyle='--')
plt.title('Nuclear mass differences in CNN-I3, CNN-I4, and LDM')
plt.xlabel("N")
plt.ylabel("Difference (MeV)")
plt.legend()
plt.xticks(ticks=range(156, 174))
plt.grid(True)
plt.tight_layout()
plt.savefig("Tests new nuclei/difference_vs_N_plot.png")
plt.close()

print('Succeeded in evaulating specific nuclei.')

# Evaluations of the CNN-I3 model on the whole dataset
model = CNN_I3().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

real_values = []
predictions = []

for idx in range(len(data)):
    z_grid, n_grid, data_feature_grid = create_5x5_neighborhood_i3(data, idx, data_feature)
    input_tensor = torch.tensor(np.array([np.stack([z_grid, n_grid, data_feature_grid])]), dtype=torch.float32).to(device)
    real_value = data.iloc[idx][data_feature]
    with torch.no_grad():
        predicted_value = model(input_tensor).item()
    real_values.append(real_value)
    predictions.append(predicted_value)

real_values = np.array(real_values)
predictions = np.array(predictions)
rmse_global = np.sqrt(np.mean((real_values - predictions) ** 2))

print(f"RMSE global I3: {rmse_global:.4f} MeV")

data['bind_ene_total_'] = data['bind_ene_total']
data['prediction_i3'] = predictions
data['difference_i3'] = real_values - predictions
data.to_csv(csv_file, index=False, sep=';')

output_file = "Tests new nuclei/differences_plot_i3_all_nuclei.png"
plot_differences(data, real_values, predictions, 
                 title="Difference between real values and predicted ones",
                 file_name=output_file)

new_nuclei_set = set(zip(new_nuclei['Z'], new_nuclei['N']))
new_nuclei_indices = data.index[data.apply(lambda row: (row['Z'], row['N']) in new_nuclei_set, axis=1)]

real_values_new = real_values[new_nuclei_indices]
predictions_new = predictions[new_nuclei_indices]

rmse_new_nuclei = np.sqrt(np.mean((real_values_new - predictions_new) ** 2))
print(f"RMSE for new nuclei I3: {rmse_new_nuclei:.4f} MeV")

output_file = "Tests new nuclei/differences_plot_i3_new_nuclei.png"
plot_differences(new_nuclei, real_values_new, predictions_new, 
                 title="Difference between real values and predicted ones",
                 file_name=output_file)


# Evaluations of the CNN-I4 on the whole dataset
model_path = "Tests new nuclei/cnn_i4_best_model_1e-05.pt"   
model = CNN_I4().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

real_values = []
predictions = []

for idx in range(len(data)):
    z_grid, n_grid, delta_I4_grid, data_feature_grid = create_5x5_neighborhood_i4(data, idx, data_feature)
    input_tensor = torch.tensor(np.array([np.stack([z_grid, n_grid, delta_I4_grid, data_feature_grid])]), dtype=torch.float32).to(device)
    real_value = data.iloc[idx][data_feature]
    with torch.no_grad():
        predicted_value = model(input_tensor).item()
    real_values.append(real_value)
    predictions.append(predicted_value)

real_values = np.array(real_values)
predictions = np.array(predictions)
rmse_global = np.sqrt(np.mean((real_values - predictions) ** 2))

print(f"RMSE global I4: {rmse_global:.4f} MeV")

data['bind_ene_total_'] = data['bind_ene_total']
data['prediction_i4'] = predictions
data['difference_i4'] = real_values - predictions
data.to_csv(csv_file, index=False, sep=';')

output_file = "Tests new nuclei/differences_plot_i4_all_nuclei.png"
plot_differences(data, real_values, predictions, 
                 title="Difference between real values and predicted ones",
                 file_name=output_file)

new_nuclei_set = set(zip(new_nuclei['Z'], new_nuclei['N']))
new_nuclei_indices = data.index[data.apply(lambda row: (row['Z'], row['N']) in new_nuclei_set, axis=1)]

real_values_new = real_values[new_nuclei_indices]
predictions_new = predictions[new_nuclei_indices]

rmse_new_nuclei = np.sqrt(np.mean((real_values_new - predictions_new) ** 2))
print(f"RMSE for new nuclei I4: {rmse_new_nuclei:.4f} MeV")

output_file = "Tests new nuclei/differences_plot_i4_new_nuclei.png"
plot_differences(new_nuclei, real_values_new, predictions_new, 
                 title="Difference between real values and predicted ones",
                 file_name=output_file)
