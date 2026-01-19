import torch
import torch.nn as nn
import torch.optim as optim
import datetime


def save_model(model, folder, best_model_state, best_test_rmse, best_epoch, num_epochs, model_name, lr_name=None):
    lr_value = lr_name if lr_name is not None else ''

    if best_model_state is not None:
            torch.save(best_model_state, f'{folder}/cnn_{model_name}_best_model_{lr_value}.pt')
            print(f'Best RMSE: {best_test_rmse:.4f}MeV found in epoch {best_epoch}')
    else:
        torch.save(model.state_dict(), f'{folder}/cnn_{model_name}_model_{num_epochs}_epochs_{lr_value}.pt')
        print('Best model not found. Saving last model')
    return


def load_model(model, device, folder, best_model_state, best_test_rmse, best_epoch, num_epochs, model_name):
    if best_model_state is not None:
        model.load_state_dict(best_model_state, map_location=device)
        print(f'Model loaded from epoch {best_epoch} with RMSE: {best_test_rmse:.4f} MeV')
    else:
        model.load_state_dict(torch.load(f'{folder}/cnn_{model_name}_model_{num_epochs}_epochs.pt', map_location=device))
        print('Best model not found. Loading last model')
    return


def train_model(model, device, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate, optimizer_name, patience, folder, model_name, lr_name=None):
    criterion = nn.MSELoss()
    OptimizerClass = getattr(optim, optimizer_name)
    optimizer = OptimizerClass(model.parameters(), lr=learning_rate)

    print(f'Total number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

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
        optimizer.zero_grad()
        train_outputs = model(train_inputs.to(device))
        train_loss = criterion(train_outputs, train_targets)
        train_loss.backward()
        optimizer.step()
        train_loss_rmse = torch.sqrt(train_loss)
        train_loss_rmse_values.append(train_loss_rmse.item())

        model.eval()
        with torch.no_grad():
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
