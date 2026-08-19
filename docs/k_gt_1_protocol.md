# K>1 Cross-Circuit Mixing Protocol

## Research Question

Can a passive website-fingerprinting attacker identify the set of websites present in a live mixed session containing K concurrently active circuits?

## Main Defense Setting

Each circuit has an opening time and emits cells according to its original relative timing. A shared bucket scheduler mixes cells from active circuits, queues overflow cells, and inserts dummy cells when a bucket has unused capacity.

## Dataset Rule

Split source trace instances into train, validation, and test partitions before mixing.

A source trace instance may appear in only one partition. Mixed training sessions use only training sources. Mixed validation sessions use only validation sources. Mixed test sessions use only test sources.

## Primary K>1 Attackers

- ARES: primary multi-label multi-tab attack
- TMWF: independent multi-tab set-prediction attack
- BAPM: block-attention multi-tab attack, if the included implementation is reproducible

DF, Tik-Tok, Var-CNN, and RF are reference attacks only. They are not primary K>1 security evidence.

## Attacker Input

The attacker receives only:

- `X`: final mixed cell-direction/timing trace
- `y`: multi-label target during training and evaluation

The attacker does not receive source indices, circuit identifiers, opening times, queue state, or dummy-cell indicators.

## Required Controls

1. Undefended concurrent traces using the same source traces and timing schedule
2. Live scheduled traces without dummy padding, if this condition is separable
3. Full live scheduler with dummy padding

## Required Reporting

- K, delta_t, N_out, and N_in
- arrival model and stagger delay
- source split version and random seed
- attack implementation and Git commit SHA
- precision, recall, F1, mAP, Precision@K, and exact-set recovery
- bandwidth overhead, completion delay, and overlap fraction
