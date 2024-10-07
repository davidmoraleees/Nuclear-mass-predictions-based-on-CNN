import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

csv_file = 'Data/mass2016_cleaned.csv'  
data = pd.read_csv(csv_file, delimiter=';')

def create_5x5_neighborhood(data, idx):
    current_n = data.iloc[idx]['N']
    current_z = data.iloc[idx]['Z']
    
    z_grid = np.zeros((5, 5))
    n_grid = np.zeros((5, 5))
    bind_ene_grid = np.zeros((5, 5))
    
    for i in range(-2, 3):
        for j in range(-2, 3):
            neighbor_n = current_n + i
            neighbor_z = current_z + j
            neighbor_idx = data[(data['N'] == neighbor_n) & (data['Z'] == neighbor_z)].index
            
            if len(neighbor_idx) > 0:
                z_grid[i+2, j+2] = neighbor_z
                n_grid[i+2, j+2] = neighbor_n
                bind_ene_grid[i+2, j+2] = data.iloc[neighbor_idx[0]]['bind_ene']
            else:
                z_grid[i+2, j+2] = 0
                n_grid[i+2, j+2] = 0
                bind_ene_grid[i+2, j+2] = 0
    
    bind_ene_grid[2, 2] = 0
    
    return z_grid, n_grid, bind_ene_grid

inputs = []
targets = []

for idx in range(len(data)):
    z_grid, n_grid, bind_ene_grid = create_5x5_neighborhood(data, idx)
    input_grid = np.stack([z_grid, n_grid, bind_ene_grid], axis=0)
    inputs.append(input_grid)
    target_value = data.iloc[idx]['bind_ene']
    targets.append(target_value)

inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.float32)
targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1)

train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs_tensor, targets_tensor, test_size=0.2, random_state=42)


class CNN_I3(nn.Module):
    def __init__(self):
        super(CNN_I3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 5 * 5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x

model = CNN_I3()
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500 
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    train_outputs = model(train_inputs)
    train_loss = criterion(train_outputs, train_targets)
    train_loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item()}')

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_inputs)
        test_loss_mse = criterion(test_outputs, test_targets)
        test_loss_rmse = torch.sqrt(test_loss_mse)
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss (RMSE): {test_loss_rmse.item()}')
    
torch.save(model.state_dict(), 'cnn_i3_model.pt')



