import numpy as np
import torch
from src.model import TaskTransformer, TaskMLP, listwise_loss, topk_rows
from src.data import Scaler, features
from src.official_scorer import score as official_score
import pandas as pd


def test_padding_permutation_gradients_and_alignment():
    torch.manual_seed(4)
    model=TaskTransformer(6,d_model=32,heads=4,layers=2,ff_dim=64,dropout=0).eval()
    x=torch.randn(1,5,6);typ=torch.tensor([[1,2,3,4,0]]);sc=torch.ones(1,5,dtype=torch.long)
    mask=torch.tensor([[1,1,1,1,0]],dtype=torch.bool);rows=torch.tensor([[11,5,7,2,-1]])
    score=model(x,typ,sc,mask)
    assert score[0,4]<-1000
    p=torch.tensor([2,0,3,1,4]);reordered=model(x[:,p],typ[:,p],sc[:,p],mask[:,p])
    torch.testing.assert_close(score[:,p],reordered,rtol=1e-5,atol=1e-5)
    chosen=topk_rows(score,rows,mask)[0].tolist()
    assert len(chosen)==3 and len(set(chosen))==3 and -1 not in chosen
    assert set(chosen)==set(topk_rows(reordered,rows[:,p],mask[:,p])[0].tolist())
    loss=listwise_loss(score,torch.tensor([1]));loss.backward()
    assert torch.isfinite(model.head[-1].weight.grad).all()
    torch.testing.assert_close(loss,torch.nn.functional.cross_entropy(score[:,:4],torch.tensor([1])))


def test_scaler_training_only():
    sc=Scaler();sc.center=np.array([1],np.float32);sc.scale=np.array([2],np.float32)
    result=sc.transform(np.array([[3],[np.nan]],np.float32))
    np.testing.assert_allclose(result,[[1,0],[0,1]])


def test_official_top3_distinct_rows():
    sol=pd.DataFrame({"sample_id":[1,2],"target":[7,2]})
    sub=pd.DataFrame({"sample_id":[1,2],"prediction":["5 5 7 1","4 6 8"]})
    assert official_score(sol,sub)==0.5


def test_feature_labels_never_enter_offline_inputs():
    d=pd.DataFrame({"sample_id":[1,1,2,2],"row_index":[0,1,0,1],
                    "target_index":[1,0,0,1],"session":["001sc1"]*4,"atco":[1]*4,
                    "timestamp":["07:00:00","07:00:00","07:00:10","07:00:10"],
                    "task_name":["assume_aircraft_A","transfer_aircraft_B"]*2,
                    "task_index":[4,5,4,5],"scenario":[1]*4,
                    "x_task_age":[1,2,11,12],"a_since_flight":[30,50,40,1],
                    "time_to_sector":[10,20,9,19]})
    a=features(d,"offline_observed")
    d.target_index=1-d.target_index
    pd.testing.assert_frame_equal(a,features(d,"offline_observed"))
    assert "target_index" not in a.columns and "atco" not in a.columns
    assert not any(c.startswith("future_") for c in features(d,"causal").columns)


def test_mlp_control_has_no_cross_task_interaction():
    model=TaskMLP(4,d_model=32,ff_dim=64,dropout=0).eval()
    x=torch.randn(1,3,4);t=torch.ones(1,3,dtype=torch.long);valid=torch.ones(1,3,dtype=torch.bool)
    with torch.no_grad():
        a=model(x,t,t,valid)
        x[:,1]+=100
        b=model(x,t,t,valid)
    torch.testing.assert_close(a[:,0],b[:,0])
    torch.testing.assert_close(a[:,2],b[:,2])
