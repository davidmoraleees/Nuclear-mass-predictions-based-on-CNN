import torch
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from models import CNN_I3, CNN_I4
from utils import create_5x5_neighborhood_i3, create_5x5_neighborhood_i4
from utils import plot_differences_new, plot_differences_combined, plot_data, evaluate_single_nucleus
from utils import fontsizes
import matplotlib.backends.backend_pdf as pdf

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

fontsizes(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
csv_file = "data/mass2020_cleaned_with_#.csv"
new_nuclei_file = "data/df2016_2020_yesyes.csv"  
model_path = "Tests new nuclei/cnn_i3_best_model_1e-05.pt"  
data_feature = config['data']['data_feature'] 

data = pd.read_csv(csv_file, delimiter=';')
new_nuclei = pd.read_csv(new_nuclei_file, delimiter=';')


# Evaluation of Mt isotopic chain
z_value = 109
n_range = range(156, 174)  
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

color_limits_storage = {}
color_limits_storage['color_limits'] = (-6, 0, 6)

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
markers = {"CNN-I3": "o", "CNN-I4": "^", "LDM": "s"}
point_size = 90

plt.figure(figsize=(10, 8))
legend_labels = set()
for model_name in ["CNN-I3", "CNN-I4", "LDM"]:
    model_data = results_df[results_df["Model"] == model_name]

    filled_points = model_data[~model_data["N"].isin([171, 172, 173])]
    plt.scatter(filled_points["N"], filled_points["Difference (MeV)"] * (-1), marker=markers[model_name], label=f"{model_name} (Training)" if f"{model_name} (Training)" not in legend_labels else None, color=colors[model_name], s=point_size, zorder=3)
    legend_labels.add(f"{model_name} (Training)")
    
    empty_points = model_data[model_data["N"].isin([171, 172, 173])]
    plt.scatter(empty_points["N"], empty_points["Difference (MeV)"] * (-1), marker=markers[model_name], label=f"{model_name} (Extrapolation)" if f"{model_name} (Extrapolation)" not in legend_labels else None, edgecolors=colors[model_name], facecolors='none', s=point_size, zorder=3)
    legend_labels.add(f"{model_name} (Extrapolation)")

    plt.plot(model_data["N"], model_data["Difference (MeV)"] * (-1), color=colors[model_name], linestyle='-', linewidth=1.5)
    # We multiply by (-1) because we are interested in nulear mass differences, not total binding energy differences.

plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel("N")
plt.ylabel("Difference (MeV)")
plt.legend()
plt.xticks(ticks=range(156, 174, 2))
plt.grid()
plt.tight_layout()
plt.savefig("Tests new nuclei/Mt_isotopic_chain.pdf")
plt.close()

print('Succeeded in evaulating the Mt isotopic chain.')


# Evaluation of N=174 isotonic chain
n_value = 174
z_range = range(110, 118)  
nuclei_to_evaluate = [(n_value, z_value) for z_value in z_range]

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
markers = {"CNN-I3": "o", "CNN-I4": "^", "LDM": "s"}
plt.figure(figsize=(10, 8))
legend_labels = set()

for model_name in ["CNN-I3", "CNN-I4", "LDM"]:
    model_data = results_df[results_df["Model"] == model_name]

    filled_points = model_data[~model_data["Z"].isin([110, 111, 112])]
    plt.scatter(filled_points["Z"], filled_points["Difference (MeV)"] * (-1), marker=markers[model_name], label=f"{model_name} (Training)" if f"{model_name} (Training)" not in legend_labels else None, color=colors[model_name], s=point_size, zorder=3)
    legend_labels.add(f"{model_name} (Training)")

    empty_points = model_data[model_data["Z"].isin([110, 111, 112])]
    plt.scatter(empty_points["Z"], empty_points["Difference (MeV)"] * (-1), marker=markers[model_name], label=f"{model_name} (Extrapolation)" if f"{model_name} (Extrapolation)" not in legend_labels else None, edgecolors=colors[model_name], facecolors='none', s=point_size, zorder=3)
    legend_labels.add(f"{model_name} (Extrapolation)")

    plt.plot(model_data["Z"], model_data["Difference (MeV)"] * (-1), color=colors[model_name], linestyle='-', linewidth=1.5)

plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel("Z")
plt.ylabel("Difference (MeV)")
plt.legend()
plt.xticks(ticks=range(110, 118, 2))
plt.grid()
plt.tight_layout()
plt.savefig("Tests new nuclei/N174_isotonic_chain.pdf")
plt.close()

print('Succeeded in evaulating the N=174 isotonic chain.')


# Evaluations of the CNN-I3 model on the whole dataset to evaluate 2020
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

output_file = "Tests new nuclei/differences_plot_i3_all_nuclei_2020.pdf"
plot_differences_new(data, real_values, predictions, file_name=output_file)

new_nuclei_set = set(zip(new_nuclei['Z'], new_nuclei['N']))
new_nuclei_indices = data.index[data.apply(lambda row: (row['Z'], row['N']) in new_nuclei_set, axis=1)]

real_values_new = real_values[new_nuclei_indices]
predictions_new = predictions[new_nuclei_indices]

rmse_new_nuclei = np.sqrt(np.mean((real_values_new - predictions_new) ** 2))
print(f"RMSE for new nuclei I3: {rmse_new_nuclei:.4f} MeV")

output_file = "Tests new nuclei/differences_plot_i3_new_nuclei_2020.pdf"
plot_differences_new(new_nuclei, real_values_new, predictions_new, file_name=output_file)
print('Succeeded in evaluating CNN-I3 on the whole dataset to evaluate 2020')


# Evaluations of the CNN-I4 on the whole dataset to evauluate 2020
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

output_file = "Tests new nuclei/differences_plot_i4_all_nuclei_2020.pdf"
plot_differences_new(data, real_values, predictions, file_name=output_file)

new_nuclei_set = set(zip(new_nuclei['Z'], new_nuclei['N']))
new_nuclei_indices = data.index[data.apply(lambda row: (row['Z'], row['N']) in new_nuclei_set, axis=1)]

real_values_new = real_values[new_nuclei_indices]
predictions_new = predictions[new_nuclei_indices]

rmse_new_nuclei = np.sqrt(np.mean((real_values_new - predictions_new) ** 2))
print(f"RMSE for new nuclei I4: {rmse_new_nuclei:.4f} MeV")

output_file = "Tests new nuclei/differences_plot_i4_new_nuclei_2020.pdf"
plot_differences_new(new_nuclei, real_values_new, predictions_new, file_name=output_file)
print('Succeeded in evaluating CNN-I4 on the whole dataset to evaluate 2020')




file = "data/mass2016_cleaned_with_#.csv"
df2016 = pd.read_csv(file, delimiter=';')

diff_ldm = df2016['Diff_bind_ene_total']


# Evaluations of the CNN-I3 model on the whole dataset to evaluate 2016
csv_file = "data/mass2016_cleaned_with_#.csv"
model_path_i3 = "Tests new nuclei/cnn_i3_best_model_1e-05.pt"  
data_feature = config['data']['data_feature'] 

data = pd.read_csv(csv_file, delimiter=';')

model_i3 = CNN_I3().to(device)
model_i3.load_state_dict(torch.load(model_path_i3, map_location=device, weights_only=True))
model_i3.eval()

real_values_i3 = []
predictions_i3 = []

for idx in range(len(data)):
    z_grid, n_grid, data_feature_grid = create_5x5_neighborhood_i3(data, idx, data_feature)
    input_tensor = torch.tensor(np.array([np.stack([z_grid, n_grid, data_feature_grid])]), dtype=torch.float32).to(device)
    real_value = data.iloc[idx][data_feature]
    with torch.no_grad():
        predicted_value = model_i3(input_tensor).item()
    real_values_i3.append(real_value)
    predictions_i3.append(predicted_value)

real_values_i3 = np.array(real_values_i3)
predictions_i3 = np.array(predictions_i3)
diff_i3 = real_values_i3 - predictions_i3
rmse_global_i3 = np.sqrt(np.mean(diff_i3 ** 2))

print(f"RMSE global I3: {rmse_global_i3:.4f} MeV")

data['bind_ene_total_antic'] = data['bind_ene_total']
data['prediction_i3_antic'] = predictions_i3
data['difference_i3_antic'] = real_values_i3 - predictions_i3
data.to_csv(csv_file, index=False, sep=';')

output_file = "Tests new nuclei/differences_plot_i3_all_nuclei_2016.pdf"
plot_differences_new(data, real_values_i3, predictions_i3, file_name=output_file)
print('Succeeded in evaluating CNN-I3 on the whole dataset to evaluate 2016')


# Evaluations of the CNN-I4 model on the whole dataset to evaluate 2016
csv_file = "data/mass2016_cleaned_with_#.csv"
model_path_i4 = "Tests new nuclei/cnn_i4_best_model_1e-05.pt"  
data_feature = config['data']['data_feature'] 

data = pd.read_csv(csv_file, delimiter=';')

model_i4 = CNN_I4().to(device)
model_i4.load_state_dict(torch.load(model_path_i4, map_location=device, weights_only=True))
model_i4.eval()

real_values_i4 = []
predictions_i4 = []

for idx in range(len(data)):
    z_grid, n_grid, delta_I4_grid, data_feature_grid = create_5x5_neighborhood_i4(data, idx, data_feature)
    input_tensor = torch.tensor(np.array([np.stack([z_grid, n_grid, delta_I4_grid, data_feature_grid])]), dtype=torch.float32).to(device)
    real_value = data.iloc[idx][data_feature]
    with torch.no_grad():
        predicted_value = model_i4(input_tensor).item()
    real_values_i4.append(real_value)
    predictions_i4.append(predicted_value)

real_values_i4 = np.array(real_values_i4)
predictions_i4 = np.array(predictions_i4)
diff_i4 = real_values_i4 - predictions_i4
rmse_global_i4 = np.sqrt(np.mean(diff_i4 ** 2))

print(f"RMSE global I4: {rmse_global_i4:.4f} MeV")

data['bind_ene_total_antic'] = data['bind_ene_total']
data['prediction_i4_antic'] = predictions_i4
data['difference_i4_antic'] = real_values_i4 - predictions_i4
data.to_csv(csv_file, index=False, sep=';')

output_file = "Tests new nuclei/differences_plot_i4_all_nuclei_2016.pdf"
plot_differences_new(data, real_values_i4, predictions_i4, file_name=output_file)
print('Succeeded in evaluating CNN-I4 on the whole dataset to evaluate 2016')


output_file_combined = "Tests new nuclei/combined_differences_plot_all_nuclei_2016.pdf"
plot_differences_combined(data, diff_i3, data, diff_i4, df2016, diff_ldm, file_name=output_file_combined)
print('Succeeded in creating the combined plot')
