import numpy as np
import torch

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
