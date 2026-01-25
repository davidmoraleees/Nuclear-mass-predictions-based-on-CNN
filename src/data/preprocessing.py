import os
import argparse
import pandas as pd
import numpy as np
from src.utils.style import fontsizes
from src.utils.plotting import plot_data
from src.utils.physics import load_constants, in_region, b_e


def rmse_print(df, metrics):
    """Print RMSE values for the provided metric columns in a dataframe."""
    for m in metrics:
        print(f"{m['label']}: {np.sqrt(np.mean(df[m['column']] ** 2))} {m['unit']}")


def load_ws4(data_folder, processed_folder, const):
    """Parse WS4.txt, clean/filter it, compute nuclear mass columns, and save WS4_cleaned.csv."""
    path = f"{data_folder}/WS4.txt"
    with open(path, "r") as f:
        for _ in range(27):
            next(f)
        header = f.readline().strip().split()
        rows = [p for line in f if (p := line.split()) and not line.startswith("=") and len(p) == len(header)]

    df = pd.DataFrame(rows, columns=header)

    A = pd.to_numeric(df["A,"], errors="coerce")
    Z = pd.to_numeric(df["Z,"], errors="coerce")
    Mth = pd.to_numeric(df["Mth"], errors="coerce")
    df = df.dropna(subset=["A,", "Z,", "Mth"]).copy()

    A, Z, Mth = A.loc[df.index].astype(int), Z.loc[df.index].astype(int), Mth.loc[df.index].astype(float)
    N = A - Z
    mask = in_region(N, Z, const)

    df = df.loc[mask].copy()
    A, Z, Mth, N = A.loc[mask], Z.loc[mask], Mth.loc[mask], N.loc[mask]

    df["N"] = N.astype(int)
    df["Mth_MeV"] = df["Mth"]
    df["Mth"] = Mth / const["uma"]
    df["atomic_mass_ws4"] = (df["Mth"] + A) * const["uma"]
    df["Z"], df["A"] = Z.astype(int), A.astype(int)
    df = df.drop(columns=["A,", "Z,"])

    df["B_e"] = b_e(df["Z"], const)
    df["M_N_ws4"] = df["atomic_mass_ws4"] - df["Z"] * const["m_e"] + df["B_e"]
    df.to_csv(f"{processed_folder}/WS4_cleaned.csv", sep=";", index=False)
    return df


def process_file(filename, header, widths, columns, names, const):
    """Read an AME fixed-width file, clean/filter it, compute LDM quantities, and return the dataframe."""
    df = pd.read_fwf(filename, usecols=columns, names=names, widths=widths, header=header, index_col=False)
    df = df.replace({"#": ""}, regex=True)

    df[["N", "Z", "A"]] = df[["N", "Z", "A"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["N", "Z", "A"]).copy()
    df = df[in_region(df["N"], df["Z"], const)]
    df[["N", "Z", "A"]] = df[["N", "Z", "A"]].astype(int)

    df["delta"] = np.where((df["Z"] % 2 == 0) & (df["N"] % 2 == 0), -1,
                           np.where(df["A"] % 2 == 1, 0,
                                    np.where((df["Z"] % 2 == 1) & (df["N"] % 2 == 1), 1, np.nan))).astype(int)
    df["delta_I4"] = ((-1) ** df["N"] + (-1) ** df["Z"]) // 2

    A, Z, N, delta = df["A"], df["Z"], df["N"], df["delta"]

    df['bind_ene'] = df['bind_ene'].astype(float) / 1000  # MeV
    df['bind_ene_total'] = df['bind_ene'] * A  # MeV
    
    df["bind_ene_teo"] = (const["av"] * A - const["aS"] * A ** (2 / 3) - const["ac"] * (Z**2) * (A ** (-1 / 3))
                          - const["aA"] * ((A - 2 * Z) ** 2) / A - const["ap"] * delta * (A ** (-1 / 2))) / A
    df["bind_ene_teo_total"] = df["bind_ene_teo"] * A

    df['Diff_bind_ene_total'] = df['bind_ene_total'] - df['bind_ene_teo_total']  # MeV

    df["A2"] = df["A2"].astype(str)
    df["atomic_mass"] = pd.to_numeric(df["A2"] + df["atomic_mass"], errors="coerce") * 1e-6 * const["uma"]
    df["atomic_mass_teo"] = Z * const["m_H"] + N * const["m_n"] - df["bind_ene_teo_total"]

    df["B_e"] = b_e(Z, const)
    df["M_N_teo"] = df["atomic_mass_teo"] - Z * const["m_e"] + df["B_e"]
    df["M_N_exp"] = df["atomic_mass"] - Z * const["m_e"] + df["B_e"]
    df["Diff_nuclear_mass"] = df["M_N_exp"] - df["M_N_teo"]
    return df


def calculate_difference(df_new, df_old, output_file):
    diff = (
        df_new.merge(df_old[["Z", "N"]], on=["Z", "N"], how="left", indicator=True)
              .query('_merge == "left_only"')
              .drop(columns="_merge")
    )
    diff.to_csv(output_file, sep=";", index=False)


def main():
    """Run the full pipeline: load constants, process AME datasets, merge WS4, compute RMSE, and plot results."""
    cfg, const = load_constants()
    fontsizes(cfg)
    images_format = cfg["training"].get("images_format", "png")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data/raw")
    parser.add_argument("--plots_folder", default="data_processing_plots")
    parser.add_argument("--processed_folder", default="data/processed")
    args = parser.parse_args()

    plots_folder = args.plots_folder
    data_folder = args.data_folder
    processed_folder = args.processed_folder

    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    ws4 = load_ws4(data_folder, processed_folder,const)

    AME = {
        2020: dict(
            file="mass2020.txt", header=28,
            columns=(1, 2, 3, 4, 6, 9, 10, 11, 13, 16, 17, 19, 21, 22),
            widths=(1, 3, 5, 5, 5, 1, 3, 4, 1, 14, 12, 13, 1, 10, 1, 2, 13, 11, 1, 3, 1, 13, 12, 1),
            names=["N-Z", "N", "Z", "A", "Element", "mass_exc", "mass_exc_unc", "bind_ene", "bind_ene_unc",
                   "beta_ene", "beta_ene_unc", "A2", "atomic_mass", "atomic_mass_unc"],
        ),
        2016: dict(
            file="mass2016.txt", header=31,
            columns=(1, 2, 3, 4, 6, 9, 10, 11, 12, 15, 16, 18, 20, 21),
            widths=[1, 3, 5, 5, 5, 1, 3, 4, 1, 13, 11, 11, 9, 1, 2, 11, 9, 1, 3, 1, 12, 11, 5],
            names=["index", "N-Z", "N", "Z", "A", "empty", "Element", "empty2", "empty3", "mass_exc", "mass_exc_unc",
                   "bind_ene", "bind_ene_unc", "empty4", "empty5", "beta_ene", "beta_ene_unc", "empty6", "A2",
                   "empty7", "atomic_mass", "atomic_mass_unc"],
        ),
    }

    years = [2016, 2020]
    dfs = {y: process_file(f"{data_folder}/{AME[y]['file']}", AME[y]["header"], AME[y]["widths"],
                        AME[y]["columns"], AME[y]["names"], const) for y in years}
    files = {y: f"{processed_folder}/mass{y}_cleaned.csv" for y in years}

    for y in years:
        dfs[y] = dfs[y].merge(ws4[["Z", "N", "M_N_ws4"]], on=["Z", "N"], how="left")
        dfs[y]["WS4_diff"] = dfs[y]["M_N_exp"] - dfs[y]["M_N_ws4"]
        dfs[y].to_csv(files[y], sep=";", index=False)

    calculate_difference(dfs[2020], dfs[2016], os.path.join(processed_folder, "ame2020_new_nuclei.csv"))

    df2016 = dfs[2016]

    rmse_print(df2016, [
        {"column": "Diff_nuclear_mass", "label": "RMSE liquid droplet model (2016) nuclear masses", "unit": "MeV"},
        {"column": "WS4_diff", "label": "RMSE WS4 nuclear masses", "unit": "MeV"},
    ])

    plot_data(df=df2016, df_column="Diff_nuclear_mass", colorbar_label=r"$\Delta$ (MeV)",
              output_path=os.path.join(plots_folder, f"nuclear_mass_expteo_dif.{images_format}"),
              cmap="seismic", vmin=-14, vcenter=0, vmax=14, title_name="LDM")

    plot_data(df=df2016, df_column="WS4_diff", colorbar_label=r"$\Delta$ (MeV)",
              output_path=os.path.join(plots_folder, f"nuclear_mass_expws4_dif.{images_format}"),
              cmap="seismic", title_name="WS4")


if __name__ == "__main__":
    main()
