## Overview
This plan covers two parallel workstreams: rebuilding the mixing simulator to operate on live, staggered circuits instead of pre-recorded complete traces, and adapting multi-label evaluation methods from the multi-tab website fingerprinting literature to properly measure the K>1 defense regime. Each workstream is broken into sequential steps that can be tracked and checked off independently.
## Workstream 1: Reimplementing Live, Staggered Mixing
### Step 1.1 — Define circuit lifecycle events
Replace the current input model (a complete `[N, T]` array of finished CW traces) with an event-driven model where each circuit has an explicit open time, a stream of real cell-generation events, and a close time. This requires converting the existing static traces into "replay streams" that emit cells at their original relative timestamps, rather than being read all at once.

- Add an `open_time` field per circuit, drawn from a configurable arrival process (fixed offset, uniform random, or exponential inter-arrival), instead of assuming all K circuits start at time 0.
- Keep the existing `cw_trace_to_packets` conversion, but change its role from "final input" to "source stream" that the scheduler pulls from incrementally.
### Step 1.2 — Convert the bucket scheduler from batch to streaming
The current `_mix_one_direction` function already implements bucketed capacity control, queueing, and dummy-fill, but it processes a fully materialized list of cells. Restructure this into a streaming scheduler that advances bucket-by-bucket and only pulls in cells from circuits whose `open_time` has already passed by that bucket's start.

- At each bucket step, check which circuits are "active" (open but not yet closed) and merge only their arrived cells into the eligible pool.
- Circuits that haven't opened yet contribute nothing — their absence is naturally covered by dummy cells, exactly as the current logic already does for empty buckets.
- Preserve the existing per-bucket real/dummy split logic (`real = eligible[:N]`, `queue = eligible[N:]`) since this part does not need to change.
### Step 1.3 — Implement staggered arrival models
Add configurable arrival patterns so experiments can compare full-overlap (all K circuits start together, the current behavior) against staggered-start conditions.

- Fixed-delay mode: second (and subsequent) circuits open a configurable number of seconds after the first, following SSBM's five-second delayed-mode design as a starting template.[^1]
- Random-delay mode: each circuit's open time is drawn from a distribution (e.g., uniform over a window) to simulate more realistic asynchronous browsing.
- Full-overlap mode: retain the current all-start-at-zero behavior as a baseline for comparison against staggered modes.
### Step 1.4 — Update output metadata for live semantics
Extend the existing output metadata (`orig_cells`, `mixed_cells`, `dummy_cells`, `delay_mean/p50/p95/max`, `bw_overhead`, `lat_overhead`) to also record each circuit's actual open/close time and its overlap fraction with other circuits in the pool, since overlap degree is known to affect both attack and defense performance significantly.[^1]

- Add `overlap_fraction` per mixed sample: the proportion of time during which more than one circuit was simultaneously active.
- Add `stagger_delay` per circuit: the time offset from the first circuit's open time.
### Step 1.5 — Validate against the offline baseline
Before trusting new results, confirm the streaming implementation produces equivalent output to the current offline version under the full-overlap, zero-delay special case (this should numerically match existing K=1 and K>1 recordings, since full-overlap is mathematically the same as the current all-at-once mixing).

- Run the new streaming scheduler with `stagger_delay = 0` for all circuits and compare `bw_overhead`/`lat_overhead` distributions against the already-recorded values for `CW_mix_K1_deltat0p01_N1/N2/N3`.
- Any mismatch indicates a bug in the streaming conversion, not a real behavioral difference, and must be resolved before moving to staggered experiments.
### Step 1.6 — Run staggered-start experiments
Once validated, sweep across stagger delay values and overlap fractions, recording bandwidth/latency overhead and downstream attack accuracy for each condition, to characterize how protection strength changes as circuits become less synchronized.
## Workstream 2: Exploring New Evaluation Methods
### Step 2.1 — Establish why single-label attacks are unsuitable
Document explicitly (already partially done in the existing report) that DF, TikTok, VarCNN, and RF are single-trace, single-website classifiers, and that applying them to a K>1 mixed trace with `groups[:,0]` as ground truth produces a meaningless label, as confirmed by the earlier overfit sanity check (2.4% accuracy on 500 training-equals-testing samples).
### Step 2.2 — Select candidate multi-label evaluation frameworks
Three existing multi-tab website fingerprinting evaluation designs are directly transferable to the K>1 cross-circuit setting, since both problems share the same underlying structure: one observed trace generated by multiple concurrently active sources.

| Method | Core idea | Fit for cross-circuit mixing |
|---|---|---|
| ARES (multi-label classification) | Treats the attack as a set of binary classifiers, one per monitored site, each predicting whether that site contributed to the mixed trace, without needing to know K in advance[^2][^3] | Best fit if the goal is "which sites are present," matching the natural multi-label nature of a K-site pool |
| BAPM (block attention profiling) | Splits the trace into blocks, uses attention to group blocks belonging to the same site, and predicts multiple site labels from a global view[^4][^5] | Useful if the report also wants to test whether a segment/position within the mixed trace can be attributed to a specific circuit |
| TMWF (set prediction) | Treats multi-tab recognition as an ordered set-prediction problem similar to object detection, using a fixed number of queries and not requiring prior knowledge of tab count[^6][^7] | Useful alternative to ARES; results are sensitive to the maximum tab/circuit count parameter, which maps naturally onto K |
### Step 2.3 — Adapt the mixed-trace labels for multi-label ground truth
Replace the current `groups[:,0]` labeling with a full multi-hot vector recording all K contributing site IDs per mixed trace, since this is the ground truth format required by all three candidate methods above.

- Modify the dataset builder (`build_mixed_dataset`) to output a `y_multilabel` array of shape `[M, num_sites]` alongside the existing `groups`/`src_indices` arrays, marking a 1 for every site present in the pool.
- Keep `groups`/`src_indices` unchanged for backward compatibility with the K=1 experiments and Option A comparisons.
### Step 2.4 — Implement or adapt one baseline multi-label attack
Start with ARES-style multi-label classification, since it is the simplest to adapt: train one binary classifier per monitored site (or a shared feature extractor with a multi-label output layer) to predict presence/absence, using standard multi-label metrics.

- Reuse existing feature extractors (DF, VarCNN) as the backbone, but replace the final softmax layer with an independent sigmoid output per site, following ARES's design of pairing a shared representation with per-site binary classifiers.[^2]
- Evaluate using multi-label metrics standard in this literature: mean average precision (mAP), precision@K, and AUC per number of contributing sites, since these are the metrics reported by ARES, BAPM, and TMWF and allow direct comparison.[^8][^2]
### Step 2.5 — Add a segment/box-based evaluation as a secondary metric
In parallel, adapt SSBM's box-based formulation, which splits a mixed trace into time segments and labels each segment with its contributing site, since this maps naturally onto the existing Δt bucket structure already used by the mixing scheduler.[^1]

- Label each bucket (or a small group of consecutive buckets) with the circuit ID(s) contributing real cells during that interval, using the metadata already tracked in Step 1.4.
- This offers a finer-grained, position-aware alternative to trace-level multi-label prediction, and can reveal whether an attacker gains an advantage by knowing which time regions are dominated by a single circuit.
### Step 2.6 — Run comparative evaluation across K, overlap, and stagger conditions
Once both evaluation methods are implemented, run them across the K and staggered-arrival configurations from Workstream 1, and report accuracy/mAP as a function of K, overlap fraction, and stagger delay, since existing multi-tab literature shows attack performance is highly sensitive to overlap degree and open-world conditions.[^7][^1]
### Step 2.7 — Cross-check results against known multi-tab attack behavior
Sanity-check new results against published multi-tab attack performance trends — for example, BAPM, TMWF, and ARES all show degrading performance as the number of concurrent tabs/circuits increases and as overlap between them decreases — to confirm the new evaluation pipeline behaves consistently with established findings before drawing conclusions about the defense's effectiveness.[^7][^8]
## Suggested Execution Order
1. Complete Steps 1.1–1.2 (event model and streaming scheduler) first, since Workstream 2 experiments are only meaningful once mixed traces reflect live, staggered timing.
2. Run Step 1.5 (validation against offline baseline) before any staggered experiments, to catch implementation bugs early.
3. Build the multi-label ground truth (Step 2.3) in parallel with Workstream 1, since it only requires a labeling change, not the streaming rewrite.
4. Implement the ARES-style multi-label baseline (Step 2.4) first, as the simplest of the three candidate methods, before attempting BAPM or TMWF-style approaches.
5. Once both workstreams are functional, run the full sweep (Steps 1.6 and 2.6) together, producing a joint dataset of overhead, timing, and multi-label attack results across all conditions.

---

## References

1. [SSBM: A spatially separated boxes-based multi-tab website ...](https://www.sciencedirect.com/science/article/abs/pii/S1084804524002005) - This paper investigates a new spatial separated boxes-based multi-tab website fingerprinting model, ...

2. [Towards Robust Multi-tab Website Fingerprinting - arXiv](https://arxiv.org/html/2501.12622v1) - Noise packets generated by multi-tab browsing and WF defenses pose significant challenges for websit...

3. [[Literature Review] Towards Robust Multi-tab Website Fingerprinting](https://www.themoonlight.io/en/review/towards-robust-multi-tab-website-fingerprinting) - ARES, an innovative framework for website fingerprinting (WF) attacks, particularly focusing on the ...

4. [Block Attention Profiling Model for Multi-tab Website Fingerprinting ...](https://dl.acm.org/doi/10.1145/3485832.3485891) - BAPM fully utilizes the whole multi-tab packet trace including the overlapping area to avoid informa...

5. [BAPM: Block Attention Profiling Model for Multi-tab Website Fingerprinting Attacks on Tor](https://dl.acm.org/doi/fullHtml/10.1145/3485832.3485891)

6. [Transformer-based Model for Multi-tab Website Fingerprinting Attack](https://github.com/jzx-bupt/TMWF) - In this paper, we propose an end-to-end multi-tab WF attack model, called Transformer-based model fo...

7. [1 Introduction](https://arxiv.org/html/2510.14283v1)

8. [DEMUX: Boundary-Aware Multi-Scale Traffic Demixing for Multi-Tab Website Fingerprinting](https://arxiv.org/html/2604.15677v1)

