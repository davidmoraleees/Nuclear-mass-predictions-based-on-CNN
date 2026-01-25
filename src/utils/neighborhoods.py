import numpy as np

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
