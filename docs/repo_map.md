# Repository Map

## Current Baseline Code

- `mix_cross_circuit.py`: original offline cross-circuit mixer
- `legacy/offline_mixer/`: frozen copy of the original mixer
- `Tamaraw.py`: Tamaraw-related baseline code
- `Recordings/`: prior experiment summaries
- `WFlib/`: included website-fingerprinting attack implementations
- `exp/`: existing train, test, preprocessing, and analysis scripts
- `scripts/`: existing WFlib model launch scripts

## New K>1 Research Code

- `src/mixer/`: live event-driven cross-circuit scheduler
- `src/data/`: source-level split and live dataset generation
- `src/adapters/`: compatible input adapters for ARES, TMWF, and BAPM
- `src/utils/`: reproducibility and I/O helpers
- `configs/`: explicit experiment settings
- `tests/`: unit, leakage, and reproducibility tests
- `manifests/`: fixed source splits and experiment metadata
- `results/summaries/`: small tracked final result files

## Local-Only Artifacts

The following directories are intentionally not committed:

- `datasets/`
- `data/raw/`
- `data/generated/`
- `checkpoints/`
- `logs/`
- `runs/`
