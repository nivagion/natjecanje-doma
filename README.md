# AWARE standalone task Transformer

**Current model:** set-wise TaskTransformer, 2 encoder layers, 4 heads, width 128, feedforward 256, dropout 0.10, 310,017 parameters. It scores every candidate task directly from original covariates. No model scores, teacher predictions, controller ID, callsign embedding, or CSV row-order position enter the network.

**GPU:** NVIDIA GeForce GTX 1660, PyTorch CUDA 12.8, verified GPU training with FP16. **Feature mode:** `offline_observed`, using future *observed covariates* within the same session. This is an offline competition experiment and is not a live operational model.

| Selected epoch by fold | In-sample outer-refit train TOP-3 by fold | Controller OOF TOP-3 | Controller macro | Worst controller | Controller SD |
|---|---|---:|---:|---:|---:|
| 4 / 4 / 20 | 87.55% / 88.25% / 99.50% | **76.60%** | 76.21% | 65.68% (001) | 4.71 pp |

The [first experiment report](reports/first_experiment.md) has the actual loss curves, controller heatmap, error tables and comparison. A same-feature per-task MLP reached **81.25%** OOF and beat this Transformer on all 16 controllers. The first Transformer is a working baseline, not a performance improvement over the historical LightGBM/003/004 systems. **Next recommended experiment:** zero-initialized gated attention residual on the stronger MLP control, with the same features, CE loss and controller folds.

![Loss and TOP-3 curves](reports/transformer_learning_curves.png)

![Controller OOF heatmap](reports/transformer_controller_heatmap.png)

**Checkpoints:** `F:\ATC_Transformer\runs\first_transformer\inner_fold{0,1,2}_best.pt` and `outer_fold{0,1,2}.pt`. Large checkpoints, raw logs and data stay on F: and are ignored by Git. **Live dashboard:** <http://127.0.0.1:8765/dashboard.html>; **TensorBoard:** <http://127.0.0.1:6006/>.

## Data and evaluation

`data_downloaded/` contains the original train, validation, test and sample submission files on F: and is ignored by Git. Training uses all 432,578 candidate rows in 11,923 snapshots from 16 controllers. Three outer folds hold out whole controllers and sessions; within every outer training partition, separate controllers choose the epoch. The selected epoch is then refit on all outer training controllers before the outer holdout is scored. The supplied validation set is not used for epoch selection. The official scorer is copied to `src/official_scorer.py` and tested.

The `offline_observed` features include future observations from the same complete session. They never use future target labels. A separate `causal` feature mode omits future observations; it has not yet been trained or evaluated.

## Windows commands

Run from this repository in PowerShell with Python 3.12.1. Large files and Python dependencies use F:. The CUDA wheel command follows [PyTorch's official version instructions](https://pytorch.org/get-started/previous-versions/).

```powershell
$env:TEMP='F:\ATC_Transformer\tmp'
$env:TMP='F:\ATC_Transformer\tmp'
$env:PIP_CACHE_DIR='F:\ATC_Transformer\cache\pip'
$env:TORCH_HOME='F:\ATC_Transformer\cache\torch'
$env:HF_HOME='F:\ATC_Transformer\cache\hf'
$env:HF_HUB_CACHE='F:\ATC_Transformer\cache\hf\hub'
py -3.12 -m venv F:\ATC_Transformer\venv
F:\ATC_Transformer\venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
F:\ATC_Transformer\venv\Scripts\python.exe -m pip install -r requirements.txt
F:\ATC_Transformer\venv\Scripts\python.exe scripts\verify_cuda.py
F:\ATC_Transformer\venv\Scripts\python.exe -m pytest -q
F:\ATC_Transformer\venv\Scripts\python.exe -m scripts.audit_data data_downloaded\train.csv data_downloaded\validation.csv data_downloaded\test.csv
F:\ATC_Transformer\venv\Scripts\python.exe -m scripts.overfit_smoke
F:\ATC_Transformer\venv\Scripts\python.exe -u -m src.train --config configs\baseline.yaml
F:\ATC_Transformer\venv\Scripts\python.exe -u -m src.train --config configs\baseline.yaml --resume
F:\ATC_Transformer\venv\Scripts\python.exe -u -m src.train --config configs\mlp_control.yaml --run-dir F:\ATC_Transformer\runs\mlp_control --resume
F:\ATC_Transformer\venv\Scripts\python.exe -m scripts.evaluate F:\ATC_Transformer\runs\first_transformer\oof_predictions.csv
F:\ATC_Transformer\venv\Scripts\python.exe -m scripts.predict --checkpoint F:\ATC_Transformer\runs\first_transformer\outer_fold0.pt --data data_downloaded\test.csv --output F:\ATC_Transformer\runs\first_transformer\test_fold0.csv
```

Open the live charts during training:

```powershell
F:\ATC_Transformer\venv\Scripts\tensorboard.exe --logdir F:\ATC_Transformer\runs\first_transformer\tensorboard --host 127.0.0.1 --port 6006
F:\ATC_Transformer\venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory F:\ATC_Transformer\runs\first_transformer
```

TensorBoard: <http://127.0.0.1:6006/>. HTML dashboard: <http://127.0.0.1:8765/dashboard.html>. It refreshes images every 15 seconds as completed epochs write real data. Run the two server commands in separate PowerShell windows if the servers are not already running.

## Architecture and protocol

The model uses train-only robust numeric scaling and missing-value indicators, task-type and scenario embeddings, a shared input projection, two self-attention layers with padding masks, and a shared ranking head. There are no positional embeddings. The baseline objective is listwise cross entropy over every real candidate. AdamW uses 0.0005 learning rate and 0.02 weight decay, cosine decay, gradient clipping at 1.0, and CUDA FP16 with scaling on this GPU. The dataset is shuffled by snapshot; rows inside a snapshot are not used as positions.

We report micro and controller-macro TOP-3, worst controller, controller spread, TOP-1, CE, coordination recall, and controller/task breakdowns. Historical numbers are marked by their original protocols in the experiment report. A second controlled loss or architecture experiment will follow the first baseline; no broad search is performed here.
