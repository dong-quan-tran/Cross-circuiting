# Attack Code Ledger

This repository already includes WFlib and its multi-tab attack implementations.

| Attack | Local implementation | Role in this project | Status |
|---|---|---|---|
| ARES | `WFlib/models/ARES.py` | Primary adaptive K>1 multi-label attack | To reproduce |
| TMWF | `WFlib/models/TMWF.py` | Independent K>1 set-prediction attack | To reproduce |
| BAPM | `WFlib/models/BAPM.py` | Block/attention-based adaptive attack | To reproduce |
| DF | `WFlib/models/DF.py` | Single-label reference baseline only | Existing |
| Tik-Tok | `WFlib/models/TikTok.py` | Single-label reference baseline only | Existing |
| Var-CNN | `WFlib/models/VarCNN.py` | Single-label reference baseline only | Existing |
| RF | `WFlib/models/RF.py` | Single-label reference baseline only | Existing |

Rules:

1. Do not edit `WFlib/models/` to make an attack fit the defense.
2. Keep cross-circuit dataset conversion in `src/adapters/`.
3. Record the Git commit SHA before every final experiment.
4. Reproduce the attack's existing baseline before evaluating cross-circuit data.
5. Treat ARES, TMWF, and BAPM as the primary K>1 attacks.
