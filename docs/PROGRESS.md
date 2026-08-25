Progress log: 08/25/2026
Today we moved the project from an **offline trace-mixing idea** into the beginning of a valid, live K>1 research pipeline.

## What we started with

The old code mixed fully completed traces after they had already been collected. That was useful for early experiments, but it did not represent real-time protection: a real observer would already have seen the unmixed traffic.

It also used standard single-site attacks and labels such as `groups[:,0]`, which do not properly fit a trace containing several websites.

## What we built

We created a new **live K>1 dataset generator**.

It now:

1. Takes raw source traces from one existing split only:
   ```text
   datasets/CW/train.npz
   datasets/CW/valid.npz
   datasets/CW/test.npz
   ```

2. Selects K traces from only that split, preventing train/test source leakage.

3. Gives circuits explicit opening times. For today’s first setting:
   ```text
   K = 2
   circuit 1 opens at 0 seconds
   circuit 2 opens at 5 seconds
   ```

4. Replays the original cells at their relative times after each circuit opens.

5. Applies a shared per-direction bucket scheduler:
   ```text
   Δt = 0.01 seconds
   N_out = N_in = 1, then swept from 1 to 5
   ```

6. Produces a correct K>1 label:
   ```text
   y.shape = [number of sessions, 95 sites]
   ```
   Each session has two active `1`s, representing the two websites present.

7. Stores private metadata separately, including source traces, open times, overlap, dummy cells, overhead, and delays. ARES never sees this metadata.

## What passed

The first K=2 smoke dataset worked correctly.

```text
X shape: (100, 10000)
y shape: (100, 95)
groups shape: (100, 2)
open times: [0.0, 5.0]
```

We also found and fixed a labeling issue: occasionally the old random source selection chose two traces from the same website, producing only one active label. The generator now requires **K distinct website labels**, so every K=2 pool now contains exactly two different sites.

```text
train: all exactly K=2 = True
valid: all exactly K=2 = True
test:  all exactly K=2 = True
```

## What failed

### 1. Naive fixed-fill padding was far too expensive

The first full design emitted exactly N cells in every bucket, filling all unused capacity with dummy cells.

At K=2, 5-second stagger, \(\Delta t=0.01\):

| N | Bandwidth overhead | Completion delay |
|---:|---:|---:|
| 1 | 2.10× | 10.53 s |
| 2 | 4.95× | 2.69 s |
| 3 | 7.88× | 1.29 s |
| 4 | 10.82× | 0.76 s |
| 5 | 13.76× | 0.46 s |

This failed the practicality goal. Increasing N fixed latency but made dummy overhead much worse.

### 2. ARES preprocessing initially could not find WFlib

Running `gen_mtaf.py` failed with:

```text
ModuleNotFoundError: No module named 'WFlib'
```

Cause: Python did not include the repository root when running a nested script.

Fix:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

MTAF feature extraction then worked.

### 3. ARES validation crashed on multi-label outputs

ARES trained successfully for one epoch using:

```text
BCEWithLogitsLoss
```

but validation crashed because WFlib sent continuous sigmoid probabilities directly into sklearn’s precision/recall/F1 functions.

Fix: locally changed the validation/test path to threshold sigmoid probabilities at 0.5 before calculating binary multi-label metrics.

After that, ARES completed two smoke-test epochs without crashing.

## Main design change

We replaced always-on fixed-fill padding with **bounded active-bucket padding**:

```text
Old full mode:
Every scheduled bucket emits N cells.
Empty slots always become dummies.

New bounded mode:
Emit real cells.
Add at most P dummy cells only when that bucket already emitted real cells.
Do not create dummy-only buckets.
```

We tested bounded padding with:

```text
P = 1 dummy cell per active bucket
```

## New result

For the same K=2, 5-second stagger, \(\Delta t=0.01\) experiment:

| N | Bounded BW overhead | Completion delay | Mean p95 cell delay |
|---:|---:|---:|---:|
| 2 | 0.132× | 2.69 s | 5.77 s |
| 3 | 0.178× | 1.29 s | 2.92 s |
| 4 | 0.209× | 0.76 s | 1.72 s |

This is the key progress today.

The bounded policy reduced overhead from:

```text
4.95× → 0.132× at N=2
7.88× → 0.178× at N=3
10.82× → 0.209× at N=4
```

while keeping the same real-cell delay as the scheduler-only control.

## Current preferred setting

Our first reasonable candidate is:

```text
K = 2
Fixed stagger = 5 seconds
Δt = 0.01 seconds
N_out = N_in = 3
Bounded mode
P = 1 dummy per active bucket
```

Its smoke-test cost is:

```text
Bandwidth overhead: 0.178×
Mean completion delay: 1.29 seconds
Mean p95 cell delay: 2.92 seconds
```

N=4 is the lower-latency alternative, but we do not yet know whether its extra overhead gives any security benefit.

## Where ARES stands

The full ARES data path now works:

```text
live mixed data
→ multi-hot labels
→ MTAF features
→ ARES on RTX A5000
→ BCEWithLogitsLoss training
```

The smoke model produced zero precision/recall/F1 after two epochs. That is expected because:

- Only 500 training sessions were used.
- There are 95 possible websites.
- The 0.5 threshold caused the young model to predict no sites.

This is not a real security result yet. It only confirms the training pipeline is working.

## Current status

We have:

- A working live K>1 simulator.
- No train/test source leakage by construction.
- Correct multi-hot labels for ARES.
- Correct source pools with distinct sites.
- Three useful controls: concurrent, scheduled, bounded.
- A practical bounded-padding candidate.
- Working MTAF and ARES GPU training path.
- A known evaluation gap: final K>1 metrics need both score-ranking metrics and thresholded set metrics.

We have **not** yet:

- Trained ARES on a full dataset.
- Evaluated attack success meaningfully.
- Selected final parameters based on security.
- Tested TMWF or BAPM.
- Generated final paper-ready results.
