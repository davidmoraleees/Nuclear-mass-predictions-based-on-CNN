import os, argparse, torch, yaml, numpy as np, pandas as pd
from src.models.cnn_i3 import CNN_I3
from src.models.cnn_i4 import CNN_I4
from src.utils.utils import create_5x5_neighborhood, plot_differences_new, plot_differences_combined, fontsizes


def main():
    p = argparse.ArgumentParser(description="Evaluate CNN models on new nuclei")
    p.add_argument("--model_path_i3", default="results_backup/cnn_i3_best_model_5e-05_prova.pt")
    p.add_argument("--model_path_i4", default="results_backup/cnn_i4_best_model_5e-05_prova.pt")
    p.add_argument("--new_nuclei_file", default="data/processed/ame2020_new_nuclei.csv")
    p.add_argument("--data_2020_file", default="data/processed/mass2020_cleaned.csv")
    p.add_argument("--data_2016_file", default="data/processed/mass2016_cleaned.csv")
    p.add_argument("--output_dir", default="tests_new_nuclei")
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    with open("config.yaml") as f: cfg = yaml.safe_load(f)
    fontsizes(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat, fmt = cfg["data"]["data_feature"], cfg["training"]["images_format"]

    new = pd.read_csv(a.new_nuclei_file, sep=";")
    new_set = set(zip(new.Z, new.N))

    def load_model(C, path):
        m = C().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m

    def evaluate_model(df, model, extras=None):
        real, pred = [], []
        for i in range(len(df)):
            z, n, *r = create_5x5_neighborhood(df, i, feat, extra_features=extras)
            x = torch.from_numpy(np.stack([z, n] + r).astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad(): pred.append(model(x).item())
            real.append(df.iloc[i][feat])
        real, pred = np.array(real), np.array(pred)
        diff = real - pred
        return real, pred, diff, np.sqrt(np.mean(diff**2))

    def rmse_subset(real, pred, mask, label):
        d = real[mask] - pred[mask]
        print(f"RMSE {label}: {np.sqrt(np.mean(d**2)):.4f} MeV")

    def new_mask(df):
        return df.apply(lambda r: (r.Z, r.N) in new_set, axis=1)

    def run(csv, year):
        df = pd.read_csv(csv, sep=";")
        for tag, C, path, ex in [
            ("I3", CNN_I3, a.model_path_i3, None),
            ("I4", CNN_I4, a.model_path_i4, ["delta_I4"]),
        ]:
            model = load_model(C, path)
            real, pred, diff, rmse = evaluate_model(df, model, ex)
            print(f"RMSE global {tag} ({year}): {rmse:.4f} MeV")
            df[f"prediction_{tag.lower()}_{year}"] = pred
            df[f"difference_{tag.lower()}_{year}"] = diff
            plot_differences_new(
                df, real, pred,
                f"{a.output_dir}/differences_plot_{tag.lower()}_all_{year}.{fmt}", tag,
            )
            if year == "2020": rmse_subset(real, pred, new_mask(df).values, f"new nuclei {tag}")
            if year == "2016": rmse_subset(real, pred, (df.Z == 50).values, f"{tag} (Z=50)")
        df.to_csv(csv, sep=";", index=False)
        return df

    d2020 = run(a.data_2020_file, "2020")
    d2016 = run(a.data_2016_file, "2016")

    df2016 = pd.read_csv(a.data_2016_file, sep=";")
    plot_differences_combined(
        d2016, d2016["difference_i3_2016"],
        d2016, d2016["difference_i4_2016"],
        df2016, df2016["Diff_bind_ene_total"],
        file_name=f"{a.output_dir}/combined_differences_plot_all_nuclei_2016.{fmt}",
    )

    print("All evaluations completed successfully.")


if __name__ == "__main__":
    main()
