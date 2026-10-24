# ECN Full-Movement Forward Model

This repository contains an ECN forward model driven by muscle spindle input,
with scripts for simulating position tuning curves and generating heatmaps.

## Quick start

```bash
python -m pip install -r requirements.txt
python run.py --out results/example --condition 8 4 --condition 1 0 --condition 8 0
python plot_results.py results/example --out results/example_figures
```

Use new output directories. `run.py` generates simulated weights and tuning
curves; `plot_results.py` writes PNG/SVG heatmaps. Add `--metrics` to the
plotting command to export selected-seed metrics as CSV.


## Model overview

- Uses 122-channel Ia/II spindle input and paw AP/ML coordinates.
- Simulates 256 cells with 32 position bins.
- Each cell samples `N` muscles, including `NI` inhibitory muscles.
- Applies a rectified linear readout to weighted spindle activity.
- Uses the same connectivity and weights for phase1 and phase3; AP is read out
  in phase1 and ML in phase3.

## Data and output

The four files in `data/228-1/` provide the recording input.
`data/real_tuning_23_sessions_32bins.npz` provides the saved real tuning
reference; the original 23-session spike trains are not included.
