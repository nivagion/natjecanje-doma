"""Nested controller-disjoint training; outer labels never select epochs."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from .data import prepare, audit, Scaler, SnapshotDataset, collate, TASK_TYPES
from .model import TaskTransformer, TaskMLP, listwise_loss
from .reporting import save_charts, write_final


def gpu_util():
    try:
        s = subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.free","--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0]
        a,b = s.split(","); return int(a), int(b)
    except Exception: return -1, -1


def loader(data, indices, scaled, batch, shuffle=False):
    return DataLoader(SnapshotDataset(data, indices, scaled), batch_size=batch, shuffle=shuffle,
                      num_workers=0, collate_fn=collate, pin_memory=True)


def run_epoch(model, batches, optimizer, amp_dtype, grad_scaler, writer, step, batch_log, fold, epoch, device):
    model.train(); loss_sum=hit3=hit1=n=0; start=time.perf_counter()
    for j, batch in enumerate(batches):
        numeric, typ, sc, valid, y, rows, ids = [v.to(device, non_blocking=True) for v in batch]
        if numeric.device.type != "cuda" or next(model.parameters()).device.type != "cuda":
            raise RuntimeError("Training batch/model not on CUDA")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            score = model(numeric, typ, sc, valid)
            loss = listwise_loss(score, y)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer); grad_scaler.update()
        with torch.no_grad():
            rank = score.argsort(dim=1, descending=True)
            hit3 += (rank[:,:3] == y[:,None]).any(dim=1).sum().item()
            hit1 += (rank[:,0] == y).sum().item()
        bs=len(y); n+=bs; loss_sum+=loss.item()*bs; step+=1
        if j % 20 == 0:
            sec=max(time.perf_counter()-start,1e-6)
            gpu, free = gpu_util()
            with open(batch_log,"a",newline="",encoding="utf8") as f:
                csv.writer(f).writerow([fold,epoch,j,step,loss.item(),n/sec,torch.cuda.memory_allocated()/1048576,gpu,free])
            writer.add_scalar(f"fold{fold}/batch_loss",loss.item(),step)
            writer.add_scalar(f"fold{fold}/throughput",n/sec,step)
    return {"loss":loss_sum/n,"top3":hit3/n,"top1":hit1/n},step


@torch.no_grad()
def evaluate(model, batches, data, device, detailed=False):
    model.eval(); loss_sum=hit3=hit1=n=0; records=[]
    for batch in batches:
        numeric, typ, sc, valid, y, rows, ids = [v.to(device, non_blocking=True) for v in batch]
        with torch.autocast("cuda", dtype=torch.float16): score=model(numeric,typ,sc,valid)
        logp=torch.log_softmax(score.float(),dim=1)
        losses=-logp.gather(1,y[:,None]).squeeze(1)
        rank=score.argsort(dim=1,descending=True)
        h3=(rank[:,:3]==y[:,None]).any(dim=1); h1=rank[:,0]==y
        bs=len(y); n+=bs; loss_sum+=losses.sum().item(); hit3+=h3.sum().item(); hit1+=h1.sum().item()
        if detailed:
            for z in range(bs):
                i=int(ids[z]); lo=int(data.starts[i]); yi=int(y[z]); top=rank[z,:3].tolist()
                records.append({"sample_id":int(data.sample_id[lo]),"controller":int(data.controller[lo]),
                                "session":str(data.session[lo]),"target_row":int(rows[z,yi]),
                                "prediction":" ".join(str(int(rows[z,t])) for t in top),
                                "hit3":int(h3[z]),"hit1":int(h1[z]),"loss":float(losses[z]),
                                "task_type":TASK_TYPES.get(int(data.task_type[lo+yi])-1,"unknown"),
                                "third_margin":float(score[z,yi]-score[z,top[-1]])})
    return {"loss":loss_sum/n,"top3":hit3/n,"top1":hit1/n}, records


def make_model(n_numeric, cfg, device):
    cls=TaskTransformer if cfg.get("model_type","transformer")=="transformer" else TaskMLP
    return cls(n_numeric, cfg["d_model"], cfg["heads"], cfg["layers"], cfg["ff_dim"],cfg["dropout"]).to(device)


def append_history(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not path.exists())


def fit(data, train_idx, val_idx, cfg, fold, run_dir, stage, epochs, patience, step, resume=False):
    scaler=Scaler().fit(data, train_idx)
    scaled=scaler.transform(data.numeric)
    train_batches=loader(data,train_idx,scaled,cfg["batch_size"],True)
    val_batches=loader(data,val_idx,scaled,cfg["batch_size"]) if val_idx is not None else None
    device=torch.device("cuda:0")
    model=make_model(scaled.shape[1],cfg,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    warmup=cfg["warmup_epochs"]
    def lr_factor(e):
        if e < warmup: return (e+1)/max(1,warmup)
        return 0.5*(1+math.cos(math.pi*(e-warmup)/max(1,epochs-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    amp_dtype=torch.bfloat16 if torch.cuda.get_device_capability(0)[0]>=8 and torch.cuda.is_bf16_supported() else torch.float16
    grad_scaler=torch.amp.GradScaler("cuda",enabled=amp_dtype==torch.float16)
    writer=SummaryWriter(str(run_dir / "tensorboard"))
    best_key=(-1,float("inf")); best_epoch=0; stale=0
    latest_path=run_dir/f"{stage}_fold{fold}_latest.pt"
    start_epoch=1
    if resume and latest_path.exists():
        saved=torch.load(latest_path,map_location="cpu",weights_only=False)
        old_cfg=dict(saved["cfg"]);new_cfg=dict(cfg)
        old_cfg.pop("batch_size",None);new_cfg.pop("batch_size",None)
        if saved["columns"]!=data.columns or old_cfg!=new_cfg: raise ValueError("Resume feature/config mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        grad_scaler.load_state_dict(saved["grad_scaler"])
        best_key=tuple(saved["best_key"]);best_epoch=saved["best_epoch"];stale=saved["stale"]
        step=saved["step"];start_epoch=saved["epoch"]+1
        torch.set_rng_state(saved["torch_rng"]);torch.cuda.set_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"]);random.setstate(saved["python_rng"])
        print(f"Resumed {stage} fold={fold} after epoch={saved['epoch']}",flush=True)
        if saved["done"]:
            writer.close();return model,scaler,best_epoch if stage=="inner" else epochs,step
    batch_log=run_dir / "batch_history.csv"
    start_all=time.perf_counter()
    for epoch in range(start_epoch,epochs+1):
        torch.cuda.reset_peak_memory_stats(); tic=time.perf_counter()
        train_metrics,step=run_epoch(model,train_batches,optimizer,amp_dtype,grad_scaler,writer,step,batch_log,fold,epoch,device)
        scheduler.step()
        val_metrics,_=evaluate(model,val_batches,data,device) if val_batches else ({"loss":float("nan"),"top3":float("nan"),"top1":float("nan")},[])
        secs=time.perf_counter()-tic; gpu,free=gpu_util()
        row={"fold":fold,"stage":stage,"epoch":epoch,"train_loss":train_metrics["loss"],"train_top3":train_metrics["top3"],
             "val_loss":val_metrics["loss"],"val_top3":val_metrics["top3"],"val_top1":val_metrics["top1"],
             "seconds":secs,"gpu_peak_mb":torch.cuda.max_memory_allocated()/1048576,"gpu_util_pct":gpu,"gpu_free_mb":free,
             "throughput":len(train_idx)/secs,"lr":optimizer.param_groups[0]["lr"]}
        append_history(run_dir/"history.csv",row)
        for key,value in row.items():
            if isinstance(value,(int,float)) and math.isfinite(value): writer.add_scalar(f"fold{fold}/{stage}/{key}",value,epoch)
        writer.flush(); save_charts(run_dir)
        print(f"fold={fold} {stage} epoch={epoch}/{epochs} train_ce={train_metrics['loss']:.4f} train@3={train_metrics['top3']:.4f} inner_ce={val_metrics['loss']:.4f} inner@3={val_metrics['top3']:.4f} sec={secs:.1f} VRAM={row['gpu_peak_mb']:.0f}MB",flush=True)
        if stage=="inner":
            key=(val_metrics["top3"],-val_metrics["loss"])
            if key>best_key:
                best_key=key;best_epoch=epoch;stale=0
                torch.save({"model":model.state_dict(),"center":scaler.center,"scale":scaler.scale,"columns":data.columns,
                            "cfg":cfg,"epoch":epoch,"fold":fold},run_dir/f"inner_fold{fold}_best.pt")
            else: stale+=1
        done=epoch==epochs or (stage=="inner" and stale>=patience)
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
               "grad_scaler":grad_scaler.state_dict(),"columns":data.columns,"cfg":cfg,"epoch":epoch,
               "best_key":best_key,"best_epoch":best_epoch,"stale":stale,"step":step,"done":done,
               "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),
               "numpy_rng":np.random.get_state(),"python_rng":random.getstate()}
        tmp=latest_path.with_suffix(".tmp");torch.save(state,tmp);os.replace(tmp,latest_path)
        if done: break
    writer.close()
    return model,scaler,best_epoch if stage=="inner" else epochs,step


def safe_fit(*args, cfg, run_dir, resume=False, **kwargs):
    while True:
        try:
            return fit(*args,cfg=cfg,run_dir=run_dir,resume=resume,**kwargs)
        except torch.cuda.OutOfMemoryError:
            if cfg["batch_size"]<=4: raise
            cfg["batch_size"]=max(4,cfg["batch_size"]//2)
            with (run_dir/"oom_adjustments.jsonl").open("a",encoding="utf8") as f:
                f.write(json.dumps({"batch_size":cfg["batch_size"],"time":time.time()})+"\n")
            torch.cuda.empty_cache();resume=True
            print(f"CUDA OOM: retrying with batch_size={cfg['batch_size']}",flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/baseline.yaml");ap.add_argument("--data",default="data_downloaded/train.csv")
    ap.add_argument("--run-dir",default="F:/ATC_Transformer/runs/first_transformer");ap.add_argument("--max-folds",type=int,default=3)
    ap.add_argument("--resume",action="store_true",help="resume saved epoch checkpoints and completed outer folds")
    args=ap.parse_args();cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf8"))
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; full training will not fall back to CPU")
    torch.set_num_threads(max(1,min(8,os.cpu_count() or 4)))
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    run_dir=Path(args.run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    (run_dir/"config.json").write_text(json.dumps(cfg,indent=2),encoding="utf8")
    df=pd.read_csv(args.data,low_memory=False);print("Data audit:",audit(df),flush=True)
    data=prepare(df,cfg["feature_mode"]);print(f"Features={len(data.columns)}, params pending; GPU={torch.cuda.get_device_name(0)}",flush=True)
    if not (run_dir/"batch_history.csv").exists():
        (run_dir/"batch_history.csv").write_text("fold,epoch,batch,step,loss,throughput,allocated_mb,gpu_util_pct,gpu_free_mb\n",encoding="utf8")
    previous=run_dir/"oof_predictions.csv"
    all_preds=pd.read_csv(previous).to_dict("records") if args.resume and previous.exists() else []
    step=0
    for fold, outer in enumerate(cfg["outer_folds"][:args.max_folds]):
        if args.resume and all_preds and fold in {int(r["fold"]) for r in all_preds}:
            print(f"Skipping completed outer fold={fold}",flush=True);continue
        outer_train=sorted(set(int(v) for v in data.controller[data.starts])-set(outer))
        inner_val=outer_train[1::4];inner_train=sorted(set(outer_train)-set(inner_val))
        if set(inner_val)&set(outer) or set(inner_train)&set(outer): raise AssertionError("Controller leakage")
        train_idx=data.indices_for(inner_train);val_idx=data.indices_for(inner_val);outer_train_idx=data.indices_for(outer_train);test_idx=data.indices_for(outer)
        (run_dir/f"fold{fold}_split.json").write_text(json.dumps({"outer":outer,"inner_train":inner_train,"inner_val":inner_val},indent=2))
        print(f"fold={fold} outer={outer} inner_val={inner_val}",flush=True)
        _,_,best_epoch,step=safe_fit(data,train_idx,val_idx,fold=fold,stage="inner",epochs=cfg["epochs"],patience=cfg["patience"],step=step,cfg=cfg,run_dir=run_dir,resume=args.resume)
        print(f"fold={fold} selected_epoch={best_epoch}; refitting on all outer-training controllers",flush=True)
        model,scaler,_,step=safe_fit(data,outer_train_idx,None,fold=fold,stage="outer_refit",epochs=best_epoch,patience=0,step=step,cfg=cfg,run_dir=run_dir,resume=args.resume)
        checkpoint={"model":model.state_dict(),"center":scaler.center,"scale":scaler.scale,"columns":data.columns,"cfg":cfg,"epoch":best_epoch,"fold":fold}
        torch.save(checkpoint,run_dir/f"outer_fold{fold}.pt")
        scaled=scaler.transform(data.numeric)
        met,records=evaluate(model,loader(data,test_idx,scaled,cfg["batch_size"]),data,torch.device("cuda:0"),True)
        for r in records:r["fold"]=fold
        all_preds.extend(records)
        pd.DataFrame(all_preds).to_csv(run_dir/"oof_predictions.csv",index=False)
        save_charts(run_dir)
        print(f"OUTER fold={fold} heldout TOP3={met['top3']:.5f} CE={met['loss']:.5f}",flush=True)
    metrics=write_final(run_dir,pd.DataFrame(all_preds));print("FINAL",json.dumps(metrics),flush=True)

if __name__=="__main__": main()
