# Data

## 228-1 model input

`228-1/` contains four NPZ files covering phase1 and phase3, with two halves
per phase.

Each file contains:

- `spindle`: `(samples, 122)`; fIa channels 0–60 and fII channels 61–121.
- `paw`: `(samples, 2)`; AP and ML coordinates in their native units.
- `spike_counts`: `(samples, 11)`; original integer counts, not used by the
  model or plotting script.

## Real tuning reference

`real_tuning_23_sessions_32bins.npz` contains the archived 23-session tuning
reference:

- `phase1_curves` and `phase3_curves`: `(32, 172)` tuning curves.
- `phase1_selected` and `phase3_selected`: boolean selection masks.
