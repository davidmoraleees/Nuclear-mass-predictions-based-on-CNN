import os, argparse, yaml, torch, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from src.models.cnn_i3 import CNN_I3
from src.models.cnn_i4 import CNN_I4
from src.utils.utils import create_5x5_neighborhood, plot_differences_nuclear_masses, plot_evolution, fontsizes
from src.training.loops import train_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["I3", "I4"], required=True)
    p.add_argument("--outputs_folder", default=None)
    a = p.parse_args()

    with open("config.yaml") as f: cfg = yaml.safe_load(f)
    fontsizes(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Model, extras, out = (
        (CNN_I3, [], "outputs_I3") if a.model_type == "I3"
        else (CNN_I4, ["delta_I4"], "outputs_I4")
    )
    out = a.outputs_folder or out
    os.makedirs(out, exist_ok=True)

    data = pd.read_csv(cfg["data"]["csv_file"], delimiter=";")
    feat = cfg["data"]["data_feature"]

    X, y = [], []
    for i in range(len(data)):
        z, n, *r = create_5x5_neighborhood(data, i, feat, extra_features=extras)
        X.append(np.stack([z, n] + r))
        y.append(data.iloc[i][feat])

    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    idx = np.arange(len(data))

    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, idx,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["general"]["random_state"],
        shuffle=True,
    )

    for lr in [1e-5, 5e-5]:
        m = Model().to(device)
        tr, te, _, best, _ = train_model(
            m, device, Xtr, ytr, Xte, yte,
            cfg["training"]["num_epochs"], lr,
            cfg["training"]["optimizer_name"],
            cfg["training"]["patience"], out,
            a.model_type, lr,
        )

        plot_evolution(tr, te, cfg["training"]["plot_skipping_epochs"],
                       cfg["training"]["num_epochs"], lr, out, a.model_type)

        plot_differences_nuclear_masses(
            data, X, y, range(len(data)), m, device,
            f"{out}/CNN-{a.model_type}_all_lr_{lr}.{cfg['training']['images_format']}",
            best,
        )
        plot_differences_nuclear_masses(
            data, Xtr, ytr, itr, m, device,
            f"{out}/CNN-{a.model_type}_train_lr_{lr}.{cfg['training']['images_format']}",
            best,
        )
        plot_differences_nuclear_masses(
            data, Xte, yte, ite, m, device,
            f"{out}/CNN-{a.model_type}_test_lr_{lr}.{cfg['training']['images_format']}",
            best,
        )


if __name__ == "__main__":
    main()
