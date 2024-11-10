import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on:', device)

csv_file = 'data/mass2016_cleaned.csv'  
data = pd.read_csv(csv_file, delimiter=';')
data_feature = 'M_N_exp'
num_epochs = 3000
learning_rate = 0.002


def create_5x5_neighborhood(data, idx, data_feature):
    current_n = data.iloc[idx]['N'] #Data for the target nucleus. 'idx'=row and 'N'=column
    current_z = data.iloc[idx]['Z']
    
    z_grid = np.zeros((5, 5))
    n_grid = np.zeros((5, 5))
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
            else:
                data_feature_grid[i + 2, j + 2] = np.nan 

    if data_feature_list:
        neighborhood_mean = np.mean(data_feature_list)
    else:
        print('Warning: data feature list is empty. Proceeding with neighborhood_mean = 0')
        neighborhood_mean = 0

    data_feature_grid[np.isnan(data_feature_grid)] = neighborhood_mean
    data_feature_grid[2, 2] = 0 #Target nucleus assigned to zero
    return z_grid, n_grid, data_feature_grid

inputs = [] #3x5x5 matrices of inputs
targets = [] #Binding energies of the target nucleus

for idx in range(len(data)):
    z_grid, n_grid, data_feature_grid = create_5x5_neighborhood(data, idx, data_feature)
    input_grid = np.stack([z_grid, n_grid, data_feature_grid], axis=0)
    inputs.append(input_grid)
    target_value = data.iloc[idx][data_feature]
    targets.append(target_value)

inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(device) #shape: (n, 3, 5, 5) where 'n' is the number of nucleus
targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1).to(device) #shape: (n, 1)

indices = np.arange(len(data)) #Original indices in the Dataframe
train_inputs, test_inputs, train_targets, test_targets, train_indices, test_indices = train_test_split(
    inputs_tensor,targets_tensor, indices, test_size=0.3, shuffle=True, random_state=42)


class CNN_I3(nn.Module):
    def __init__(self):
        super(CNN_I3, self).__init__() #We initialize the nn.Module class
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) #Basic features
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) #More complex features
        self.fc = nn.Linear(32 * 5 * 5, 1) #Number of input features=32*5*5=800. Output features=1.
        self.relu = nn.ReLU()

    def forward(self, x): #'x' is the input
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x)) #We have a 4D tensor: (batch size, channels, height, width).
        x = x.view(x.size(0), -1) #The fc requires a 2D tensor: (batch size, features). batch_size=x.size(0)
        x = self.fc(x)
        return x

def save_model(model, best_model_state, best_test_rmse, best_epoch, num_epochs):
    if best_model_state is not None:
            torch.save(best_model_state, f'cnn_i3_best_model.pt')
            print(f'Best RMSE: {best_test_rmse:.4f}MeV found in epoch {best_epoch}')
    else:
        torch.save(model.state_dict(), f'cnn_i3_model_{num_epochs}_epochs.pt')
        print('Best model not found. Saving last model')
    return


def load_model(model, best_model_state, best_test_rmse, best_epoch, num_epochs):
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Model loaded from epoch {best_epoch} with RMSE: {best_test_rmse:.4f} MeV')
    else:
        model.load_state_dict(torch.load(f'cnn_i3_model_{num_epochs}_epochs.pt'))
        print('Best model not found. Loading last model')
    return


def train_model(model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate):
    criterion = nn.MSELoss() #Instance of the MSE
    optimizer = optim.Adamax(model.parameters(), lr=learning_rate) #model.parameters()=weights and biases to optimize
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

    save_model(model, best_model_state, best_test_rmse, best_epoch, num_epochs)
    return train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch


model = CNN_I3().to(device) #Instance of our model
train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch = train_model(
    model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate)


plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_loss_rmse_values, label='Training RMSE', color='blue', linewidth=0.5)
plt.plot(range(1, num_epochs + 1), test_loss_rmse_values, label='Test RMSE', color='red', linewidth=0.5)
plt.title(f'Evolution of RMSE over {num_epochs} epochs')
plt.xlabel('Època')
plt.ylabel('RMSE (MeV)')
max_value = max(max(train_loss_rmse_values), max(test_loss_rmse_values)) + 1
plt.ylim(0, max_value) 
plt.legend()
plt.grid()
plt.savefig(f'CNN-I3 plots/CNN-I3_evolution.png')
plt.show()



def plot_differences(data, inputs, targets, indices, model, device, title, file_name):
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
        color_limits_storage['color_limits'] = (vmin, vmax)
    else:
        vmin, vmax = color_limits_storage['color_limits']

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
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
    plt.title(title)
    plt.savefig(file_name)
    plt.show()
    return


color_limits_storage = {}
plot_differences(data, inputs_tensor, targets_tensor, range(len(data)), model, device,
                'Difference exp-predicted (all data)', 'CNN-I3 plots/CNN-I3_diff_scatter.png')

plot_differences(data, train_inputs, train_targets, train_indices, model, device,
                'Difference exp-predicted (training set)', 'CNN-I3 plots/CNN-I3_diff_scatter_train.png')

plot_differences(data, test_inputs, test_targets, test_indices, model, device,
                'Difference exp-predicted (test set)', 'CNN-I3 plots/CNN-I3_diff_scatter_test.png')


n_splits = 5  
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

rmse_train_list = []
rmse_test_list = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs_tensor)):
    print(f"Fold {fold + 1}/{n_splits}")

    train_inputs, test_inputs = inputs_tensor[train_idx], inputs_tensor[test_idx]
    train_targets, test_targets = targets_tensor[train_idx], targets_tensor[test_idx]
    model = CNN_I3().to(device)

    train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate)
    
    rmse_train_list.append(min(train_loss_rmse_values))
    rmse_test_list.append(best_test_rmse) 

    print(f"Fold {fold + 1}: Best RMSE (Test): {best_test_rmse:.4f}MeV in epoch {best_epoch}")

mean_rmse_train = np.mean(rmse_train_list)
std_rmse_train = np.std(rmse_train_list)
mean_rmse_test = np.mean(rmse_test_list)
std_rmse_test = np.std(rmse_test_list)

print(f"Final Average Train RMSE: {mean_rmse_train:.4f} ± {std_rmse_train:.4f} MeV")
print(f"Final Average Test RMSE: {mean_rmse_test:.4f} ± {std_rmse_test:.4f} MeV")