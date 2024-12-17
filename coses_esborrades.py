# One training of the model
model = CNN_I3().to(device) #Instance of our model
train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch_test = train_model(
    model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate, optimizer_name, patience, I3_results_folder, model_name)

plot_evolution(train_loss_rmse_values, test_loss_rmse_values, plot_skipping_epochs, num_epochs, learning_rate, I3_results_folder, model_name)

color_limits_storage = {}
plot_differences(data, inputs_tensor, targets_tensor, range(len(data)), model, device,
                'Difference exp-predicted (all data)', f'{I3_results_folder}/CNN-I3_diff_scatter.png', best_test_rmse)

plot_differences(data, train_inputs, train_targets, train_indices, model, device,
                'Difference exp-predicted (training set)', f'{I3_results_folder}/CNN-I3_diff_scatter_train.png', best_test_rmse)

plot_differences(data, test_inputs, test_targets, test_indices, model, device,
                'Difference exp-predicted (test set)', f'{I3_results_folder}/CNN-I3_diff_scatter_test.png', best_test_rmse)

# Now we convert total binding energy predictions into nuclear mass predictions
color_limits_storage = {}
plot_differences_nuclear_masses(data, inputs_tensor, targets_tensor, range(len(data)), model, device,
                                'Difference exp-predicted (all data) nuclear masses', f'{I3_results_folder}/CNN-I3_diff_scatter_nuclear_masses.png', best_test_rmse)

plot_differences_nuclear_masses(data, train_inputs, train_targets, train_indices, model, device,
                                'Difference exp-predicted (training set) nuclear masses', f'{I3_results_folder}/CNN-I3_diff_scatter_train_nuclear_masses.png', best_test_rmse)

plot_differences_nuclear_masses(data, test_inputs, test_targets, test_indices, model, device,
                                'Difference exp-predicted (test set) nuclear masses', f'{I3_results_folder}/CNN-I3_diff_scatter_test_nuclear_masses.png', best_test_rmse)


# K-folding
n_splits = config['kfolding']['n_splits']
kf = KFold(n_splits=n_splits, shuffle=True, random_state=config['general']['random_state'])

rmse_train_list = []
rmse_test_list = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs_tensor)):
    print(f"Fold {fold + 1}/{n_splits}")

    train_inputs, test_inputs = inputs_tensor[train_idx], inputs_tensor[test_idx]
    train_targets, test_targets = targets_tensor[train_idx], targets_tensor[test_idx]
    model = CNN_I3().to(device)

    train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch_test = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate, optimizer_name, patience, I3_results_folder, model_name)
    
    rmse_train_list.append(min(train_loss_rmse_values))
    rmse_test_list.append(best_test_rmse) 

    print(f"Fold {fold + 1}: Best RMSE (Test): {best_test_rmse:.4f}MeV in epoch {best_epoch_test}")

mean_rmse_train = np.mean(rmse_train_list)
std_rmse_train = np.std(rmse_train_list)
mean_rmse_test = np.mean(rmse_test_list)
std_rmse_test = np.std(rmse_test_list)

print(f"Final Average Train RMSE: {mean_rmse_train:.4f} ± {std_rmse_train:.4f} MeV")
print(f"Final Average Test RMSE: {mean_rmse_test:.4f} ± {std_rmse_test:.4f} MeV")









# One training of the model
model = CNN_I4().to(device) #Instance of our model
train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch_test = train_model(
    model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate, optimizer_name, patience, I4_results_folder, model_name)

plot_evolution(train_loss_rmse_values, test_loss_rmse_values, plot_skipping_epochs, num_epochs, learning_rate, I4_results_folder, model_name)

color_limits_storage = {}
plot_differences(data, inputs_tensor, targets_tensor, range(len(data)), model, device,
                'Difference exp-predicted (all data)', f'{I4_results_folder}/CNN-I4_diff_scatter.png', best_test_rmse)

plot_differences(data, train_inputs, train_targets, train_indices, model, device,
                'Difference exp-predicted (training set)', f'{I4_results_folder}/CNN-I4_diff_scatter_train.png', best_test_rmse)

plot_differences(data, test_inputs, test_targets, test_indices, model, device,
                'Difference exp-predicted (test set)', f'{I4_results_folder}/CNN-I4_diff_scatter_test.png', best_test_rmse)

# Now we convert total binding energy predictions into nuclear mass predictions
color_limits_storage = {}
plot_differences_nuclear_masses(data, inputs_tensor, targets_tensor, range(len(data)), model, device,
                                'Difference exp-predicted (all data) nuclear masses', f'{I4_results_folder}/CNN-I4_diff_scatter_nuclear_masses.png', best_test_rmse)

plot_differences_nuclear_masses(data, train_inputs, train_targets, train_indices, model, device,
                                'Difference exp-predicted (training set) nuclear masses', f'{I4_results_folder}/CNN-I4_diff_scatter_train_nuclear_masses.png', best_test_rmse)

plot_differences_nuclear_masses(data, test_inputs, test_targets, test_indices, model, device,
                                'Difference exp-predicted (test set) nuclear masses', f'{I4_results_folder}/CNN-I4_diff_scatter_test_nuclear_masses.png', best_test_rmse)


# K-folding
n_splits = config['kfolding']['n_splits']
kf = KFold(n_splits=n_splits, shuffle=True, random_state=config['general']['random_state'])

rmse_train_list = []
rmse_test_list = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs_tensor)):
    print(f"Fold {fold + 1}/{n_splits}")

    train_inputs, test_inputs = inputs_tensor[train_idx], inputs_tensor[test_idx]
    train_targets, test_targets = targets_tensor[train_idx], targets_tensor[test_idx]
    model = CNN_I4().to(device)

    train_loss_rmse_values, test_loss_rmse_values, num_epochs, best_test_rmse, best_epoch_test = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, num_epochs, learning_rate, optimizer_name, patience, I4_results_folder, model_name)
    
    rmse_train_list.append(min(train_loss_rmse_values))
    rmse_test_list.append(best_test_rmse) 

    print(f"Fold {fold + 1}: Best RMSE (Test): {best_test_rmse:.4f}MeV in epoch {best_epoch_test}")

mean_rmse_train = np.mean(rmse_train_list)
std_rmse_train = np.std(rmse_train_list)
mean_rmse_test = np.mean(rmse_test_list)
std_rmse_test = np.std(rmse_test_list)

print(f"Final Average Train RMSE: {mean_rmse_train:.4f} ± {std_rmse_train:.4f} MeV")
print(f"Final Average Test RMSE: {mean_rmse_test:.4f} ± {std_rmse_test:.4f} MeV")
