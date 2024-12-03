import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.model_selection import train_test_split
import os

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class CNN_I3(nn.Module):
    def __init__(self):
        super(CNN_I3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=config['model']['conv1']['kernel_size'],
                               stride=config['model']['conv1']['stride'], padding=config['model']['conv1']['padding'])
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=config['model']['conv2']['kernel_size'],
                               stride=config['model']['conv2']['stride'], padding=config['model']['conv2']['padding'])
        self.fc = nn.Linear(32 * 5 * 5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class CNN_I4(nn.Module):
    def __init__(self):
        super(CNN_I4, self).__init__() #We initialize the nn.Module class
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=config['model']['conv1']['kernel_size'],
                                stride=config['model']['conv1']['stride'], padding=config['model']['conv1']['padding']) #Basic features
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=config['model']['conv2']['kernel_size'],
                                stride=config['model']['conv2']['stride'], padding=config['model']['conv2']['padding']) #More complex features
        self.fc = nn.Linear(32 * 5 * 5, 1) #Number of input features=32*5*5=800. Output features=1.
        self.relu = nn.ReLU()

    def forward(self, x): #'x' is the input
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x)) #We have a 4D tensor: (batch size, channels, height, width).
        x = x.view(x.size(0), -1) #The fc requires a 2D tensor: (batch size, features). batch_size=x.size(0)
        x = self.fc(x)
        return x

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
csv_file = "data/mass2020_cleaned_with_#.csv"
new_nuclei_file = "data/df2016_2020_yesyes.csv"  
model_path = "Tests new nuclei/cnn_i3_best_model_0.001.pt"  
data_feature = config['data']['data_feature'] 

data = pd.read_csv(csv_file, delimiter=';')
new_nuclei = pd.read_csv(new_nuclei_file, delimiter=';')

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
