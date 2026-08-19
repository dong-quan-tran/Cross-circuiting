# Experiment Manifests

Every generated dataset and training run must have a small manifest file.

Required fields:

- experiment ID
- Git commit SHA
- source dataset name and checksum
- source split version
- random seed
- K
- delta_t
- N_out and N_in
- arrival mode
- stagger parameter
- number of mixed train, validation, and test examples
- attack model and its configuration
- hardware and software environment
- output dataset location
- output result location

The raw dataset, checkpoints, and full logs stay local on the lab computer.
Only configurations and small final result summaries should be pushed to GitHub.
