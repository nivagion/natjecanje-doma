"""Training charts, controller audit, and live-refresh dashboard."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summary(pred: pd.DataFrame) -> dict:
    hit3 = pred.hit3.astype(float)
    group = pred.groupby("controller").hit3.mean()
    coord=pred.task_type.isin(["entry_coordination","exit_coordination"])
    return {"micro_top3":float(hit3.mean()), "macro_top3":float(group.mean()),
            "worst_top3":float(group.min()), "controller_std":float(group.std(ddof=0)),
            "p10_top3":float(np.percentile(group,10)), "p25_top3":float(np.percentile(group,25)),
            "bottom_quartile_top3":float(group.nsmallest(max(1,len(group)//4)).mean()),
            "gap_top3":float(group.max()-group.min()), "micro_top1":float(pred.hit1.mean()),
            "loss":float(pred.loss.mean()), "coordination_recall":float(pred.loc[coord,"hit3"].mean()),"n":int(len(pred))}


def save_charts(run_dir: Path):
    hist_path = run_dir / "history.csv"
    if not hist_path.exists(): return
    h = pd.read_csv(hist_path)
    if h.empty: return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for fold, part in h.groupby("fold"):
        inner = part[part.stage == "inner"]
        if inner.empty: continue
        axes[0,0].plot(inner.epoch, inner.train_loss, label=f"train {fold}")
        axes[0,0].plot(inner.epoch, inner.val_loss, linestyle="--", label=f"inner val {fold}")
        axes[0,1].plot(inner.epoch, inner.train_top3, label=f"train {fold}")
        axes[0,1].plot(inner.epoch, inner.val_top3, linestyle="--", label=f"inner val {fold}")
        best=inner.sort_values(["val_top3","val_loss"],ascending=[False,True]).iloc[0]
        axes[0,1].scatter([best.epoch],[best.val_top3],marker="*",s=95)
        axes[1,0].plot(inner.epoch, inner.val_top1, label=f"inner val {fold}")
        axes[1,1].plot(inner.epoch, inner.gpu_peak_mb, label=f"peak MB {fold}")
    for ax, title in zip(axes.flat,["Cross entropy", "TOP-3", "Inner TOP-1", "Peak VRAM MB"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run_dir / "learning_curves.png", dpi=130); plt.close(fig)
    batch_path=run_dir/"batch_history.csv"
    if batch_path.exists():
        b=pd.read_csv(batch_path)
        if len(b):
            fig,ax=plt.subplots(figsize=(11,3))
            for fold,part in b.groupby("fold"):
                ax.plot(part.step,part.loss.rolling(20,min_periods=1).mean(),label=f"fold {fold}")
            ax.set(title="Smoothed training batch CE",xlabel="Optimizer step",ylabel="CE")
            ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(run_dir/"batch_loss.png",dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3))
    for fold,part in h.groupby("fold"):
        part=part[part.stage=="inner"]
        if part.empty:continue
        for ax,col,title in zip(axes,["gpu_util_pct","throughput","seconds"],["GPU utilization %","Snapshots/s","Seconds/epoch"]):
            ax.plot(part.epoch,part[col],label=f"fold {fold}");ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(run_dir/"gpu_training.png",dpi=130);plt.close(fig)
    pred_path = run_dir / "oof_predictions.csv"
    if pred_path.exists():
        p = pd.read_csv(pred_path)
        if len(p):
            mat = p.groupby(["fold", "controller"]).hit3.mean().unstack()
            fig, ax = plt.subplots(figsize=(12, 2.5 + .55*len(mat)))
            im = ax.imshow(mat.to_numpy(), vmin=.5, vmax=1, cmap="RdYlGn")
            ax.set_xticks(range(len(mat.columns)), [f"{int(c):03d}" for c in mat.columns])
            ax.set_yticks(range(len(mat.index)), [f"fold {i}" for i in mat.index])
            for (y,x), val in np.ndenumerate(mat.to_numpy()):
                if np.isfinite(val): ax.text(x,y,f"{val:.1%}",ha="center",va="center",fontsize=8)
            fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(run_dir / "controller_heatmap.png", dpi=130); plt.close(fig)
            metrics=[]
            for fold,part in p.groupby("fold"):
                acc=part.groupby("controller").hit3.mean()
                metrics.append((fold,acc.mean(),acc.min(),acc.std(ddof=0),acc.max()-acc.min(),acc.nsmallest(max(1,len(acc)//4)).mean()))
            m=pd.DataFrame(metrics,columns=["fold","macro","worst","std","gap","bottom_quartile"])
            fig,ax=plt.subplots(figsize=(10,4))
            for c in m.columns[1:]:ax.plot(m.fold,m[c],marker="o",label=c)
            ax.set(title="Actual outer-held-out controller metrics",xlabel="Completed outer fold",ylabel="Rate or spread")
            ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(run_dir/"fairness.png",dpi=130);plt.close(fig)
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>AWARE Transformer live</title><style>body{font-family:system-ui;max-width:1200px;margin:auto;background:#101923;color:#eaf1f6}img{max-width:100%;background:white}a{color:#9ce}</style></head><body><h1>AWARE Transformer training</h1><p>Images refresh every 15 seconds. Only completed epochs and held-out outer folds are shown.</p><img src="learning_curves.png"><img src="batch_loss.png"><img src="gpu_training.png"><h2>Controller OOF TOP-3</h2><img src="controller_heatmap.png"><img src="fairness.png"><p><a href="history.csv">Epoch history</a> · <a href="batch_history.csv">Batch history</a> · <a href="oof_predictions.csv">OOF predictions</a></p><script>setInterval(()=>document.querySelectorAll('img').forEach(img=>{const base=img.src.split('?')[0];img.src=base+'?t='+Date.now()}),15000)</script></body></html>'''
    (run_dir / "dashboard.html").write_text(html, encoding="utf8")


def write_final(run_dir: Path, pred: pd.DataFrame):
    pred.to_csv(run_dir / "oof_predictions.csv", index=False)
    metrics = summary(pred)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    per = pred.groupby("controller").agg(n=("hit3","size"), top3=("hit3","mean"), top1=("hit1","mean"), loss=("loss","mean"))
    per.to_csv(run_dir / "controller_metrics.csv")
    pred.groupby("task_type").agg(n=("hit3","size"), top3=("hit3","mean"), top1=("hit1","mean")).to_csv(run_dir / "task_type_metrics.csv")
    pred.groupby(["controller","task_type"]).agg(n=("hit3","size"), top3=("hit3","mean"), errors=("hit3",lambda s:int((1-s).sum()))).to_csv(run_dir/"controller_task_errors.csv")
    edges=[-np.inf,-2,-1,-.5,-.1,0,.1,.5,1,2,np.inf]
    pred.assign(margin_bin=pd.cut(pred.third_margin,edges)).groupby("margin_bin",observed=True).agg(n=("hit3","size"),top3=("hit3","mean")).to_csv(run_dir/"boundary_metrics.csv")
    save_charts(run_dir)
    return metrics
