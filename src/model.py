"""Permutation-equivariant Transformer ranking every candidate in a snapshot."""
import torch
from torch import nn


class TaskTransformer(nn.Module):
    def __init__(self, n_numeric: int, d_model=128, heads=4, layers=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.type_embedding = nn.Embedding(32, 16, padding_idx=0)
        self.scenario_embedding = nn.Embedding(32, 8, padding_idx=0)
        self.input = nn.Sequential(nn.Linear(n_numeric + 24, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_dim, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, numeric, task_type, scenario, valid):
        x = torch.cat((numeric, self.type_embedding(task_type), self.scenario_embedding(scenario)), dim=-1)
        x = self.attention(self.input(x), src_key_padding_mask=~valid)
        return self.head(x).squeeze(-1).masked_fill(~valid, -1e4)


class TaskMLP(nn.Module):
    """Per-task control with identical inputs and no cross-candidate exchange."""
    def __init__(self,n_numeric,d_model=128,heads=4,layers=2,ff_dim=256,dropout=0.1):
        super().__init__()
        self.type_embedding=nn.Embedding(32,16,padding_idx=0)
        self.scenario_embedding=nn.Embedding(32,8,padding_idx=0)
        self.input=nn.Sequential(nn.Linear(n_numeric+24,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout))
        self.blocks=nn.Sequential(*[nn.Sequential(nn.Linear(d_model,ff_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(ff_dim,d_model),nn.LayerNorm(d_model)) for _ in range(layers)])
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1))

    def forward(self,numeric,task_type,scenario,valid):
        x=torch.cat((numeric,self.type_embedding(task_type),self.scenario_embedding(scenario)),dim=-1)
        x=self.blocks(self.input(x))
        return self.head(x).squeeze(-1).masked_fill(~valid,-1e4)


def listwise_loss(scores, target):
    return nn.functional.cross_entropy(scores.float(), target)


def topk_rows(scores, rows, valid, k=3):
    scores = scores.masked_fill(~valid, -1e4)
    indices = scores.topk(min(k, scores.size(1)), dim=1).indices
    return rows.gather(1, indices)
