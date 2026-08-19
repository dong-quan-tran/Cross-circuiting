# Lab Machine Setup

## One-Time Setup

```bash
git clone https://github.com/dong-quan-tran/Cross-circuiting.git
cd Cross-circuiting
git checkout main
bash scripts/setup_lab_env.sh
```

## Confirm GPU Access

```bash
source .venv/bin/activate
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected GPU: NVIDIA RTX A5000.

## Daily Workflow

```bash
cd Cross-circuiting
git checkout main
git pull origin main
source .venv/bin/activate
```

## Local-Only Directories

Do not commit these directories:

```text
datasets/
data/raw/
data/generated/
checkpoints/
logs/
runs/
```
