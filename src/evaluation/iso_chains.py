import os, argparse, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.models.cnn_i3 import CNN_I3
from src.models.cnn_i4 import CNN_I4
from src.utils.style import fontsizes
from src.utils.neighborhoods import create_5x5_neighborhood
from src.utils.evaluation import evaluate_single_nucleus
from src.utils.config import load_config


def main():
    p = argparse.ArgumentParser(description="Evaluate isotopic and isotonic chains")
    p.add_argument("--input-data", default="data/processed/mass2020_cleaned.csv")
    p.add_argument("--new-nuclei-file", default="data/processed/ame2020_new_nuclei.csv")
    p.add_argument("--i3-model", default="results_backup/cnn_i3_best_model_1e-05.pt")
    p.add_argument("--i4-model", default="results_backup/cnn_i4_best_model_1e-05.pt")
    p.add_argument("--output-dir", default="tests_iso_chains")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    cfg = load_config()
    fontsizes(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_fmt = cfg["training"]["images_format"]

    CHAINS = [
        dict(name="isotonic", fixed=174, rng=range(110,118), x="Z", key="Z",
             title="N=174 isotonic chain", extrap=[110,111,112], xt=range(110,118),
             xlim=None, out=f"N174_isotonic_chain.{img_fmt}"),
        dict(name="isotopic", fixed=109, rng=range(156,174), x="N", key="N",
             title="Mt isotopic chain (Z=109)", extrap=[171,172,173], xt=range(156,176,2),
             xlim=(155,175), out=f"Mt_isotopic_chain.{img_fmt}")
    ]

    data = pd.read_csv(a.input_data, sep=";")
    new = pd.read_csv(a.new_nuclei_file, sep=";")
    new_set = set(zip(new.N, new.Z))

    feat = cfg["data"]["data_feature"]
    av,aS,ac,aA,ap = (cfg["LDM"][k] for k in ["av","aS","ac","aA","ap"])

    models = [
        dict(n="I3", cls=CNN_I3, p=a.i3_model, ex=[]),
        dict(n="I4", cls=CNN_I4, p=a.i4_model, ex=["delta_I4"]),
    ]

    colors = {"I3":"blue","I4":"red","LDM":"green","WS4":"purple"}
    markers = {"I3":"o","I4":"^","LDM":"s","WS4":"v"}
    ps = 130

    for c in CHAINS:
        nuclei = [(c["fixed"], z) for z in c["rng"]] if c["key"]=="Z" else [(n, c["fixed"]) for n in c["rng"]]
        res = []

        for m in models:
            net = m["cls"](cfg).to(device)
            net.load_state_dict(torch.load(m["p"], map_location=device, weights_only=True))
            for N,Z in nuclei:
                try:
                    r,p,d = evaluate_single_nucleus(data, net, N, Z, feat, create_5x5_neighborhood, extra_features=m["ex"])
                    res.append(dict(N=N,Z=Z,**{"Real Value (MeV)":r,"Predicted Value (MeV)":p,"Difference (MeV)":d,"Model":m["n"]}))
                except ValueError: pass

        for N,Z in nuclei:
            A = N+Z
            d = -1 if (Z%2==0 and N%2==0) else 0 if A%2 else 1
            ldm = av*A - aS*A**(2/3) - ac*Z**2*A**(-1/3) - aA*(A-2*Z)**2/A - ap*d*A**(-1/2)
            r = data.loc[(data.N==N)&(data.Z==Z), feat].values
            if r.size:
                res.append(dict(N=N,Z=Z,**{"Real Value (MeV)":r[0],"Predicted Value (MeV)":ldm,"Difference (MeV)":r[0]-ldm,"Model":"LDM"}))

        for N,Z in nuclei:
            r = data.loc[(data.N==N)&(data.Z==Z), feat].values
            w = data.loc[(data.N==N)&(data.Z==Z), "WS4_diff"].values
            if r.size and w.size:
                res.append(dict(N=N,Z=Z,**{"Real Value (MeV)":r[0],"Difference (MeV)":-w[0],"Model":"WS4"}))

        df = pd.DataFrame(res)
        df["In 2020?"] = df.apply(lambda x:(x.N,x.Z) in new_set, axis=1)
        df.to_csv(os.path.join(a.output_dir,"predictions_nuclei.csv"), sep=";", index=False)

        plt.figure(figsize=(10,6))
        seen = set()
        y = -df["Difference (MeV)"]
        ymin,ymax = np.floor(y.min()), np.ceil(y.max())

        for n in ["I3","I4","LDM","WS4"]:
            d = df[df.Model==n]
            tr = d[~d[c["key"]].isin(c["extrap"])]
            ex = d[d[c["key"]].isin(c["extrap"])]
            plt.scatter(tr[c["key"]],-tr["Difference (MeV)"],marker=markers[n],color=colors[n],s=ps,
                        label=f"{n} (Training)" if n not in seen else None)
            plt.scatter(ex[c["key"]],-ex["Difference (MeV)"],marker=markers[n],edgecolors=colors[n],
                        facecolors="none",s=ps,label=f"{n} (Extrapolation)" if n not in seen else None)
            plt.plot(d[c["key"]],-d["Difference (MeV)"],color=colors[n],lw=1.5)
            seen.add(n)

        plt.axhline(0,color="black",lw=2.5)
        plt.xlabel(c["x"]); plt.ylabel(r"$\Delta$ (MeV)"); plt.title(c["title"])
        plt.xticks(c["xt"]); plt.ylim(ymin-0.5,ymax+0.5)
        if c["xlim"]: plt.xlim(*c["xlim"])
        plt.grid()
        plt.legend(handles=[Line2D([0],[0],marker=markers[k],color=colors[k],linestyle="None",label=k)
                            for k in ["I3","I4","LDM","WS4"]], loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(a.output_dir,c["out"]), bbox_inches="tight")
        plt.close()
        print(f"Succeeded in evaluating the {c['name']} chain.")


if __name__ == "__main__":
    main()
