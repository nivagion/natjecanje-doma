"""Score all candidates with a saved checkpoint, export official row_index strings."""
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data import prepare, Scaler, SnapshotDataset, collate
from src.model import TaskTransformer, TaskMLP

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--data",required=True);ap.add_argument("--output",required=True);args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for inference in this command")
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    df=pd.read_csv(args.data,low_memory=False)
    if "target_index" not in df:df["target_index"]=0
    # prepare's one-target audit is for training; test has no labels. Supply one dummy target per snapshot.
    if not df.groupby("sample_id").target_index.sum().eq(1).all():
        first=~df.sample_id.duplicated();df.loc[first,"target_index"]=1
    data=prepare(df,ck["cfg"]["feature_mode"])
    if data.columns!=ck["columns"]:raise ValueError("Feature schema mismatch")
    scaler=Scaler();scaler.center=ck["center"];scaler.scale=ck["scale"]
    scaled=scaler.transform(data.numeric)
    cls=TaskTransformer if ck["cfg"].get("model_type","transformer")=="transformer" else TaskMLP
    model=cls(scaled.shape[1],ck["cfg"]["d_model"],ck["cfg"]["heads"],ck["cfg"]["layers"],ck["cfg"]["ff_dim"],ck["cfg"]["dropout"]).to("cuda:0")
    model.load_state_dict(ck["model"]);model.eval();records=[]
    batches=DataLoader(SnapshotDataset(data,np.arange(len(data.starts)),scaled),batch_size=64,collate_fn=collate)
    with torch.no_grad():
        for numeric,typ,sc,valid,y,rows,ids in batches:
            score=model(numeric.cuda(),typ.cuda(),sc.cuda(),valid.cuda())
            top=score.argsort(dim=1,descending=True)[:,:3].cpu().numpy()
            for j,idx in enumerate(ids.numpy()):
                selected=[int(rows[j,k]) for k in top[j] if int(rows[j,k])>=0]
                if len(set(selected))!=3:raise AssertionError("Invalid TOP-3")
                records.append({"sample_id":int(data.sample_id[data.starts[idx]]),"prediction":" ".join(map(str,selected))})
    pd.DataFrame(records).to_csv(args.output,index=False)
    print(f"wrote {len(records)} predictions to {args.output}")

if __name__=="__main__":main()
