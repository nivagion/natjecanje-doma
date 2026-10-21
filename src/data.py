"""Label-independent snapshot features and controller-safe preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

FORBIDDEN = {"sample_id", "row_index", "target_index", "session", "atco", "timestamp", "task_name", "task_index", "scenario", "seq_len"}
RELATIVE = ["time_to_sector", "x_task_age", "x_state_age", "k_timeToExitPoint", "a_since_any", "a_since_flight", "time_to_MD", "distance_to_zone", "k_cfl_minus_fl"]
TASK_TYPES = {0:"entry_conflict_resolution",1:"exit_conflict_resolution",2:"entry_coordination",3:"exit_coordination",4:"assume_aircraft",5:"transfer_aircraft",6:"conflict_resolution",7:"non_conformance_resolution",8:"qos_improvement",9:"entry_conditions",10:"exit_conditions",12:"zone_conflict",13:"return_to_route"}


def audit(df: pd.DataFrame, labelled: bool = True) -> dict:
    g = df.groupby("sample_id", sort=False)
    sizes = g.size()
    if not df.groupby("sample_id")["atco"].nunique().eq(1).all():
        raise ValueError("A snapshot crosses controllers")
    if not df.groupby("session")["atco"].nunique().eq(1).all():
        raise ValueError("A session crosses controllers")
    if df.duplicated(["sample_id", "row_index"]).any():
        raise ValueError("Duplicate candidate row_index")
    if labelled and not g.target_index.sum().eq(1).all():
        raise ValueError("Exactly one target per snapshot required")
    return {"rows":len(df), "snapshots":len(sizes), "controllers":df.atco.nunique(), "sessions":df.session.nunique(), "min_candidates":int(sizes.min()), "max_candidates":int(sizes.max())}


def features(df: pd.DataFrame, mode: str = "offline_observed") -> pd.DataFrame:
    if mode not in {"offline_observed", "causal"}:
        raise ValueError(mode)
    # The target is deliberately removed before any feature engineering.
    d = df.drop(columns=["target_index"], errors="ignore").copy()
    cols = [c for c in d.columns if c not in FORBIDDEN and pd.api.types.is_numeric_dtype(d[c])]
    x = d[cols].astype("float32").copy()
    g = d.groupby("sample_id", sort=False)
    x["candidate_count"] = g.row_index.transform("size").astype("float32")
    x["task_type_count"] = d.groupby([d.sample_id, d.task_index], sort=False).row_index.transform("size").astype("float32")
    # Aircraft ID is only a grouping key, never a predictive category.
    task_type_name=d.task_index.map(TASK_TYPES).fillna("unknown")
    aircraft=pd.Series([str(name)[len(kind)+1:].split("_")[0] if str(name).startswith(kind+"_") else str(name).split("_")[-1]
                        for name,kind in zip(d.task_name,task_type_name)],index=d.index)
    x["aircraft_task_count"] = d.groupby([d.sample_id, aircraft], sort=False).row_index.transform("size").astype("float32")
    for c in RELATIVE:
        if c not in d:
            continue
        a = g[c]
        x[c + "_pct"] = a.rank(pct=True).astype("float32")
        x[c + "_from_min"] = (d[c] - a.transform("min")).astype("float32")
        x[c + "_from_median"] = (d[c] - a.transform("median")).astype("float32")
    if mode == "offline_observed":
        # Observed later snapshots in the same session. This is an offline feature.
        seconds = pd.to_timedelta(d.timestamp).dt.total_seconds().astype("float32")
        h = pd.DataFrame({"session":d.session, "task_name":d.task_name, "seconds":seconds,
                          "sample_id":d.sample_id, "task_age":d.x_task_age}, index=d.index)
        h = h.sort_values(["session", "seconds", "sample_id"], kind="stable")
        key = [h.session, h.task_name]
        task = h.groupby(key, sort=False)
        h["future_seen_seconds"] = task.seconds.shift(-1) - h.seconds
        h["future_last_seconds"] = task.seconds.transform("max") - h.seconds
        h["future_observations"] = task.cumcount(ascending=False).astype("float32")
        h["future_age_change"] = task.task_age.shift(-1) - h.task_age
        for c in ["future_seen_seconds", "future_last_seconds", "future_observations", "future_age_change"]:
            x[c] = h[c].reindex(d.index).astype("float32")
        # Future observed action-clock reset, previously useful in offline references.
        ac = pd.DataFrame({"sample_id":d.sample_id,"session":d.session,"seconds":seconds,
                           "aircraft":aircraft,"age":d.a_since_flight}, index=d.index)
        action_cols=[c for c in ["a_last_assume_control","a_last_force_assume_control",
                 "a_last_assign_level_clearance","a_last_assign_lateral_clearance",
                 "a_last_initiate_transfer_of_control"] if c in d]
        for c in action_cols: ac[c] = d[c]
        ac=ac.groupby(["sample_id","session","seconds","aircraft"],sort=False,as_index=False).min()
        ac=ac.sort_values(["session","seconds"],kind="stable")
        key=[ac.session,ac.aircraft]
        prev=ac.groupby(key,sort=False).age.shift()
        reset=ac.age.lt(prev-1)
        action_time=(ac.seconds-ac.age).where(reset)
        ac["future_action_time"]=action_time.groupby(key,sort=False).transform(lambda z:z.shift(-1).bfill())
        ac["future_action_delay"]=ac.future_action_time-ac.seconds
        for c in action_cols:
            event=ac[c].where(reset)
            ac["future_"+c]=event.groupby(key,sort=False).transform(lambda z:z.shift(-1).bfill())
        lookup=ac.set_index(["sample_id","aircraft"])[["future_action_delay"]+["future_"+c for c in action_cols]]
        mapped=lookup.reindex(pd.MultiIndex.from_arrays([d.sample_id,aircraft]))
        mapped.index=d.index
        for c in mapped: x[c]=mapped[c].astype("float32")
    if FORBIDDEN.intersection(x.columns) or any("pred" in c.lower() for c in x):
        raise ValueError("Forbidden model input")
    return x.replace([np.inf, -np.inf], np.nan)


@dataclass
class Prepared:
    numeric: np.ndarray
    task_type: np.ndarray
    scenario: np.ndarray
    row_index: np.ndarray
    sample_id: np.ndarray
    controller: np.ndarray
    session: np.ndarray
    target: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    columns: list[str]

    def indices_for(self, controllers: list[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.controller[self.starts], controllers))


def prepare(df: pd.DataFrame, mode: str) -> Prepared:
    audit(df)
    df = df.sort_values(["sample_id", "row_index"], kind="stable").reset_index(drop=True)
    x = features(df, mode)
    sid = df.sample_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1]
    ends = np.r_[starts[1:], len(df)]
    target = np.flatnonzero(df.target_index.to_numpy() == 1)
    assert len(target) == len(starts)
    return Prepared(x.to_numpy(dtype=np.float32), df.task_index.to_numpy(dtype=np.int64) + 1,
                    df.scenario.to_numpy(dtype=np.int64) + 1, df.row_index.to_numpy(dtype=np.int64),
                    sid, df.atco.to_numpy(dtype=np.int64), df.session.to_numpy(), target - starts,
                    starts, ends, list(x.columns))


class Scaler:
    def fit(self, data: Prepared, snapshot_indices: np.ndarray):
        rows = np.concatenate([np.arange(data.starts[i], data.ends[i]) for i in snapshot_indices])
        a = data.numeric[rows]
        with np.errstate(all="ignore"):
            self.center = np.nanmedian(a, axis=0).astype(np.float32)
            self.center = np.nan_to_num(self.center)
            self.scale = np.nanpercentile(a, 75, axis=0) - np.nanpercentile(a, 25, axis=0)
            self.scale = np.nan_to_num(self.scale, nan=1, posinf=1, neginf=1).astype(np.float32)
            self.scale[self.scale < 1e-5] = 1
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(a)
        values = np.nan_to_num((a - self.center) / self.scale, nan=0, posinf=0, neginf=0)
        return np.concatenate([np.clip(values, -8, 8), missing.astype(np.float32)], axis=-1).astype(np.float32)


class SnapshotDataset:
    def __init__(self, data: Prepared, indices: np.ndarray, scaled: np.ndarray):
        self.data, self.indices, self.scaled = data, indices, scaled
    def __len__(self): return len(self.indices)
    def __getitem__(self, j):
        i = self.indices[j]; lo, hi = self.data.starts[i], self.data.ends[i]
        return (self.scaled[lo:hi], self.data.task_type[lo:hi],
                self.data.scenario[lo:hi], self.data.target[i], self.data.row_index[lo:hi], i)


def collate(batch):
    import torch
    b, max_len = len(batch), max(len(v[1]) for v in batch)
    x = np.zeros((b, max_len, batch[0][0].shape[1]), np.float32)
    task = np.zeros((b, max_len), np.int64); scenario = np.zeros_like(task)
    mask = np.zeros((b, max_len), bool); rows = np.full((b, max_len), -1, np.int64)
    target = np.empty(b, np.int64); indices = np.empty(b, np.int64)
    for j, (num, typ, sc, y, ri, i) in enumerate(batch):
        n = len(typ); x[j,:n] = num; task[j,:n] = typ; scenario[j,:n] = sc
        mask[j,:n] = True; rows[j,:n] = ri; target[j] = y; indices[j] = i
    return tuple(torch.from_numpy(a) for a in (x, task, scenario, mask, target, rows, indices))
