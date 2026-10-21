"""Align small historical controller summary with this run's held-out OOF."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument("predictions");args=ap.parse_args()
    p=Path(args.predictions);cur=pd.read_csv(p).groupby("controller").hit3.agg(["size","mean"])
    cur.columns=["n","transformer"]
    h=pd.read_csv("reports/historical_controller_top3.csv").set_index("model").T
    h.index=h.index.astype(int)
    h=h.rename(columns={"nitro_typed320":"lightgbm_typed320","meta_top10_0.25":"reference_003",
                        "blend_meta_top10_0.25_probability_0.75":"reference_004"})
    out=cur.join(h)
    for col in h:out["delta_"+col]=out.transformer-out[col]
    out.to_csv(p.parent/"historical_controller_comparison.csv")
    mat=out[["transformer","lightgbm_typed320","reference_003","reference_004"]].T
    fig,ax=plt.subplots(figsize=(13,3.8))
    im=ax.imshow(mat.to_numpy(),vmin=.5,vmax=1,cmap="RdYlGn")
    ax.set_xticks(range(len(mat.columns)),[f"{int(c):03d}" for c in mat.columns])
    ax.set_yticks(range(len(mat.index)),["Transformer","LightGBM","003","004"])
    for (y,x),val in np.ndenumerate(mat.to_numpy()):ax.text(x,y,f"{val:.0%}",ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(p.parent/"historical_comparison.png",dpi=140);plt.close(fig)
    print(out.round(4).to_string())

if __name__=="__main__":main()
