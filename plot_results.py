"""Saved-output-only metrics and real-versus-simulation position heatmaps."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

PHASES = (('phase1', 'AP (paw col 9)'), ('phase3', 'ML (paw col 11)'))
REAL_TUNING = Path(__file__).resolve().parent / 'data' / 'real_tuning_23_sessions_32bins.npz'


def load_run(source):
    source = Path(source)
    config = json.loads((source / 'run_config.json').read_text())
    if config['bins'] != 32:
        raise ValueError('The bundled real tuning reference has 32 bins')
    matrices, masks = {}, {}
    with np.load(REAL_TUNING, allow_pickle=False) as saved:
        for phase, _ in PHASES:
            arr = saved[f'{phase}_curves']
            mask = saved[f'{phase}_selected']
            if (arr.shape != (32, 172) or mask.shape != (172,)
                    or mask.dtype != np.dtype(bool)):
                raise ValueError(f'{REAL_TUNING}: invalid {phase} curves or MI mask')
            matrices[phase] = arr
            masks[phase] = mask
    return config, matrices, masks


def load_chunk(path, bins):
    with np.load(path, allow_pickle=False) as saved:
        w = saved['weights']
        curves = {p: saved[p] for p, _ in PHASES}
    if w.shape != (122, 256) or not np.isfinite(w).all():
        raise ValueError(f'{path}: invalid weights')
    for phase, arr in curves.items():
        if arr.shape != (bins, 256) or np.any(arr[np.isfinite(arr)] < 0):
            raise ValueError(f'{path}: invalid {phase} curve')
    return curves


def peaks(matrix, min_modulation=1.0):
    """Contiguous-plateau-center peaks on raw curves; -1 means ineligible."""
    matrix = np.asarray(matrix, dtype=float)
    indices = np.full(matrix.shape[1], -1, dtype=int)
    for cell, values in enumerate(matrix.T):
        if (not np.isfinite(values).all() or np.ptp(values) == 0
                or np.ptp(values) < min_modulation):
            continue
        maxima = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-8))
        if len(maxima) and np.all(np.diff(maxima) == 1):
            indices[cell] = int(np.floor(np.mean(maxima)))
    return indices


def load_display(source, config, seed):
    """Read only the selected seed; other seed chunks are not needed for plotting."""
    if seed < 0 or seed not in config['seeds']:
        raise ValueError('Selected seed is not in this run')
    return {tuple(pair): load_chunk(Path(source) /
            f'n{pair[0]:02d}_i{pair[1]:02d}_s{seed:02d}.npz', config['bins'])
            for pair in config['conditions']}


def outer_peaks(peak, bins):
    """Outer 20% by bin centers: equal left/right counts, including at 32 bins."""
    peak = np.asarray(peak)
    position = (peak + .5) / bins
    return (peak >= 0) & ((position < .2) | (position > .8))


def summarize(source, out, sim_min_modulation=2.0, seed=6, metrics=False):
    if not np.isfinite(sim_min_modulation) or sim_min_modulation < 0:
        raise ValueError('Simulated modulation threshold must be finite and nonnegative')
    config, real, masks = load_run(source)
    display = load_display(source, config, seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bins = config['bins']
    rows = []
    if metrics:
        for (n, ni), matrices in display.items():
            for phase, _ in PHASES:
                curve = matrices[phase]
                peak = peaks(curve, sim_min_modulation)
                size = curve.shape[1]
                n_good = int((peak >= 0).sum())
                n_outer = int(outer_peaks(peak, bins).sum())
                rows.append({'n': n, 'ni': ni, 'seed': seed, 'size': size, 'axis': phase,
                             'n_eligible': n_good, 'n_outer': n_outer,
                             'fraction_outer_all': n_outer / size,
                             'fraction_outer_eligible': n_outer / n_good if n_good else '',
                             'fraction_eligible': n_good / size})
        with (out / 'selected_seed_metrics.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_comparison(real, display, masks, bins, out, sim_min_modulation, seed)
    return out


def sorted_panel(raw, min_modulation=0, selected=None):
    """Gate/sort raw curves, then smooth (sigma=1 bin) and row-z for display."""
    raw = np.asarray(raw, dtype=float)
    order = peaks(raw, min_modulation)
    eligible = order >= 0
    if selected is not None:
        selected = np.asarray(selected)
        if selected.shape != eligible.shape or selected.dtype != np.dtype(bool):
            raise ValueError('Expected one boolean selection per unit')
        eligible &= selected
    keep = np.flatnonzero(eligible)
    keep = keep[np.argsort(order[keep], kind='stable')]
    matrix = np.asarray(raw)[:, keep].T
    if not len(keep):
        return matrix
    smooth = gaussian_filter1d(matrix, 1, axis=1, mode='nearest')
    mean = smooth.mean(axis=1, keepdims=True)
    sd = smooth.std(axis=1, keepdims=True)
    return np.divide(smooth - mean, sd, out=np.full_like(smooth, np.nan), where=sd > 0)


def plot_comparison(real, simulated, masks, bins, out, sim_min_modulation, seed):
    """Row-z heatmaps only, peak-sorted on unsmoothed raw rate."""
    labels = [('Real (MI-selected)', real)] + [(f'N={n}, NI={ni}', curves)
                                            for (n, ni), curves in simulated.items()]
    fig, axes = plt.subplots(2, len(labels), squeeze=False,
                             figsize=(max(6, 2.3 * len(labels)), 6),
                             constrained_layout=True)
    for row, (phase, axis_label) in enumerate(PHASES):
        for col, (label, curves) in enumerate(labels):
            raw = curves[phase]
            z = sorted_panel(raw, 0 if col == 0 else sim_min_modulation,
                             masks[phase] if col == 0 else None)
            ax = axes[row, col]
            if len(z):
                ax.imshow(np.ma.masked_invalid(z), aspect='auto', origin='lower',
                          interpolation='none', cmap='bwr', vmin=-2, vmax=2)
            else:
                ax.text(.5, .5, 'No eligible cells', transform=ax.transAxes, ha='center')
            ax.set_xticks([0, (bins - 1) // 2, bins - 1],
                          labels=['1', str((bins - 1) // 2 + 1), str(bins)])
            ax.set_title(f'{label}\n{len(z)} units', fontsize=10)
            ax.set_xlabel(f'Position bin (1-{bins})', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{axis_label.split()[0]}\nPeak-sorted unit', fontsize=10)
            ax.tick_params(labelsize=9)
    n_real = real['phase1'].shape[1]
    fig.suptitle(f'Real: 23 sessions, {n_real} units | Model input: 228-1 | Seed {seed}\n'
                 f'Simulated peak-to-trough modulation >= {sim_min_modulation:g} Hz',
                 fontsize=11)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(-2, 2), cmap='bwr'),
        ax=axes.ravel().tolist(), fraction=.025, pad=.02, ticks=[-2, 0, 2])
    colorbar.set_label('z-score', fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    fig.savefig(out / 'real_vs_sim_heatmaps.png', dpi=200)
    fig.savefig(out / 'real_vs_sim_heatmaps.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Completed simulation run')
    parser.add_argument('--out', required=True, type=Path, help='New figure directory')
    parser.add_argument('--seed', type=int, default=6, help='Saved simulation seed to display (default: 6)')
    parser.add_argument('--sim-min-modulation-hz', type=float, default=2.0)
    parser.add_argument('--metrics', action='store_true', help='Export selected-seed 256-cell metrics')
    args = parser.parse_args()
    if (args.seed < 0 or not np.isfinite(args.sim_min_modulation_hz)
            or args.sim_min_modulation_hz < 0):
        parser.error('Seed and simulated modulation threshold must be finite and nonnegative')
    print('Saved:', summarize(args.source, args.out, args.sim_min_modulation_hz,
                              args.seed, args.metrics))


if __name__ == '__main__':
    main()
