"""Full-movement ECN simulation driven by bundled 228-1 spindle and paw data."""

import argparse
import json
from pathlib import Path

import numpy as np

from data import full_blocks, load_session
from model import muscle_clusters, relu_rate, weights
from readout import fit_ranges, readout

DEFAULT_CONDITIONS = ((8, 4), (1, 0), (8, 0), (16, 4), (8, 2),
                      (16, 6), (16, 8), (16, 0), (16, 10))
MASTER_SEED = 20261004


def portable(value):
    """JSON-safe arrays, including missing-bin NaNs."""
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [portable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(portable(value), indent=2, allow_nan=False) + '\n')


def conditions(pairs):
    pairs = [tuple(p) for p in pairs]
    if (not pairs or len(set(pairs)) != len(pairs)
            or any(not (1 <= n <= 61 and 0 <= ni <= n) for n, ni in pairs)):
        raise ValueError('Conditions must be distinct pairs with 1 <= N <= 61, 0 <= NI <= N')
    return pairs


def run(out, bins=32, pairs=DEFAULT_CONDITIONS, seeds=10, seed=None):
    pairs = conditions(pairs)
    if bins != 32 or seeds < 1 or (seed is not None and seed < 0):
        raise ValueError('This project uses 32 bins; seeds must be positive')
    seed_ids = list(range(seeds)) if seed is None else [seed]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'run_config.json', {
        'conditions': pairs, 'bins': bins, 'seeds': seed_ids, 'master_seed': MASTER_SEED})

    _, halves = load_session()
    blocks, block_audit = full_blocks(halves)
    ranges = fit_ranges(halves, blocks, bins)
    labels, pairing, clusters = muscle_clusters(halves, blocks)
    write_json(out / 'model_info.json', {'blocks': block_audit, 'ranges': ranges,
              'cluster_labels': labels, 'cluster_pairing': pairing, 'cluster_info': clusters})
    for n, ni in pairs:
        for seed in seed_ids:
            w, _ = weights(n, ni, 256, MASTER_SEED + seed, labels, pairing)
            curves, _ = readout(halves, blocks, ranges,
                lambda h, a, b: relu_rate(halves[h]['x'][a:b], w) * .01, bins)
            np.savez_compressed(out / f'n{n:02d}_i{ni:02d}_s{seed:02d}.npz',
                                weights=w, phase1=curves['phase1'], phase3=curves['phase3'])
            print(f'N={n} NI={ni} seed={seed}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New, nonexistent run directory')
    parser.add_argument('--condition', action='append', nargs=2, type=int, metavar=('N', 'NI'))
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument('--seeds', type=int, default=None, help='Run seeds 0 through count-1')
    seed_options.add_argument('--seed', type=int, default=6,
                              help='Run only this seed index (default: 6)')
    args = parser.parse_args()
    pairs = conditions(args.condition or DEFAULT_CONDITIONS)
    if ((args.seeds is not None and args.seeds < 1)
            or (args.seed is not None and args.seed < 0) or args.out.exists()):
        parser.error('Seeds must be positive and output directory must not already exist')
    seed = None if args.seeds is not None else args.seed
    seeds = args.seeds if args.seeds is not None else 10
    print('Saved:', run(args.out, pairs=pairs, seeds=seeds, seed=seed))


if __name__ == '__main__':
    main()
