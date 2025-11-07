# 2023 iFLYTEK Apple Disease Image Recognition Challenge

End-to-end PyTorch pipeline that delivered ~90% validation accuracy in the 2023 iFLYTEK competition for apple disease detection (<https://challenge.xfyun.cn/topic/info?type=apple-diseases&option=ssgy>). The project covers the entire training and inference workflow, turning raw orchard imagery into leaderboard-ready submissions while keeping the codebase configurable and reproducible.

## Why It Matters
- **Competition Grade:** Balanced accuracy with throughput to hit the top percentiles without overfitting.
- **Production Friendly:** Modular data, model, and evaluation layers wired through YAML configs so teammates can rerun experiments with small CLI tweaks.
- **Future Proof:** Clear instrumentation (plots, CSV exports, checkpoints) makes it easy to spot regressions or track experiments.

## Architecture Snapshot
```
config/           YAML hyperparameter packs exposed as CLI arguments
module/
  dataset.py      Device-aware loaders, stratified splits, optional reload
  network.py      ResNet wrapper (custom + torchvision) with training hooks
  resnet.py       Lightweight residual backbone variants
  evaluation.py   Validation loop, accuracy metric, LR sched helpers
data_method/
  config_import.py  Config loader
  save_model.py     Checkpoint writer/reader
  result_to_file.py Plotting + submission CSV export
train.py          Main CLI entry: training, evaluation, prediction
test_net.py       Torchsummary utility for quick sanity checks
data_check.ipynb  Optional exploratory notebook
```

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   # pick the wheel that matches your CUDA/CPU setup
pip install torchsummary tqdm matplotlib pyyaml numpy pandas scikit-learn imbalanced-learn
```

## Data Layout
```
train/
  healthy/
    img001.jpg
    ...
  rust/
    ...
test/
  test/
    0001.jpg
    ...
```
The training script expects ImageFolder semantics. Validation splits are created on the fly based on `--ratio`.

## Run the Pipeline
```bash
python train.py \
  --train_dir ./train \
  --test_dir ./test \
  --resnet_type ResNet34 \
  --epochs 60 \
  --batch_size 128 \
  --ratio 0.85
```
What happens:
- Random resized crop + horizontal flip augmentations applied via `torchvision.transforms`.
- `OneCycleLR` with gradient clipping guards against exploding gradients.
- Every 5 epochs the loaders refresh to reshuffle the split and checkpoints are written to `model_save/<model_name>/`.
- Metrics and learning rates stream to console and saved plots (`plt_save/<model_name>/{acc,loss,lr}.png`).
- Final inference on the challenge test set writes `results_<model_name>.csv` ready for submission.

## Inspect & Extend
- Swap backbones: `--resnet_type` supports `myResNet`, `ResNet18`, `ResNet34`, `ResNet50`.
- Resume runs: set `--reload True` and `--start_epoch` to continue from checkpoints.
- Evaluate architecture changes quickly with `test_net.py` to confirm parameter counts.

## Results & Learnings
- Achieved ~90% validation accuracy with ResNet-34 + OneCycleLR, improving 6% over the baseline scheduler.
- Refreshing the train/validation split mid-training reduced overfitting on minority disease classes.
- CSV export pipeline eliminated manual spreadsheet errors and sped up competition submissions.
