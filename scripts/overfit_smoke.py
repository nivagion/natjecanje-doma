"""Deliberately fit a tiny in-sample subset to catch implementation errors."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data import prepare, Scaler, SnapshotDataset, collate
from src.model import TaskTransformer, listwise_loss

def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    d=prepare(pd.read_csv("data_downloaded/train.csv",low_memory=False),"offline_observed")
    ix=np.arange(32);sc=Scaler().fit(d,ix);scaled=sc.transform(d.numeric)
    model=TaskTransformer(scaled.shape[1]).to("cuda:0")
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    batch=next(iter(DataLoader(SnapshotDataset(d,ix,scaled),batch_size=32,collate_fn=collate)))
    numeric,typ,scenario,valid,y,*_=[v.cuda() for v in batch]
    first=None
    for step in range(80):
        model.train();opt.zero_grad(set_to_none=True)
        scores=model(numeric,typ,scenario,valid);loss=listwise_loss(scores,y);loss.backward();opt.step()
        if first is None:first=float(loss.detach())
    model.eval()
    with torch.no_grad():
        scores=model(numeric,typ,scenario,valid)
        last=float(listwise_loss(scores,y));top3=float((scores.topk(3,dim=1).indices==y[:,None]).any(dim=1).float().mean())
    print({"first_ce":first,"final_ce":last,"train_top3":top3,"steps":80,"device":str(numeric.device)})
    if last>=first*.65:raise AssertionError("Tiny subset did not overfit")

if __name__=="__main__":main()
