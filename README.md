# Cross-Circuit Mixing for Website Fingerprinting Defense

This repository studies cross-circuit mixing as a website fingerprinting defense in the K>1 setting.

The project evaluates whether an attacker can recover the set of websites present in one live mixed session containing multiple concurrently active circuits.

## Current Research Direction

1. Split raw source traces into train, validation, and test partitions before mixing.
2. Replay K circuits with live and staggered opening times.
3. Schedule cells through shared per-direction time buckets.
4. Compare concurrent traffic alone, scheduling without padding, and scheduling with dummy padding.
5. Evaluate using multi-label multi-tab attackers:
   - ARES (primary)
   - TMWF (secondary)
   - BAPM (secondary)

## Repository Structure

- `WFlib/`: included WF attack models and shared utilities.
- `exp/`: WFlib training, testing, and feature-extraction scripts.
- `src/`: new live mixer, dataset builder, and K>1 evaluation code.
- `scripts/`: launchers for source splitting, data generation, and attacks.
- `tests/`: correctness, no-leakage, and reproducibility tests.
- `legacy/`: archived offline cross-circuit scripts.

## Data Policy

Datasets, generated traces, model checkpoints, and logs remain local to the lab machine and are excluded from Git.

## Legacy Work

Tamaraw and one-page experiments are maintained separately in:
https://github.com/dong-quan-tran/Website-Fingerprinting-Tamaraw-trade-off-in-One-page-setting-Phase-2-Summer-2026-REU