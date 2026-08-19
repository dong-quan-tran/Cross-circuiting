# New K>1 Research Code

This folder contains new code for the live, staggered cross-circuit mixing study.

- `data/`: source-level splitting, dataset validation, and live dataset generation
- `mixer/`: event-driven circuit replay, bucket scheduler, padding, and metrics
- `adapters/`: conversion from cross-circuit datasets to the existing WFlib attack input format
- `utils/`: reproducibility and file utilities

Do not modify WFlib model implementations directly for this study.
Keep the external attack implementations stable and adapt the generated cross-circuit datasets to their expected input format.
