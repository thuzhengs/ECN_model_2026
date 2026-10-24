"""Load the bundled real 228-1 NPZ archive."""

from pathlib import Path

import numpy as np

PHASES = ('phase1', 'phase3')
HALVES = ('phase1_half1', 'phase1_half2', 'phase3_half1', 'phase3_half2')
DEFAULT_DATA = Path(__file__).resolve().parent / 'data' / '228-1'


def load_session(path=DEFAULT_DATA):
    """Return neuron IDs and all four halves: paw [AP, ML], spikes, spindle x."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'{path}: expected a curated NPZ directory')
    halves, ids = {}, None
    for name in HALVES:
        file = path / f'{name}.npz'
        with np.load(file, allow_pickle=False) as saved:
            if (str(saved['version'].item()) != 'v2_raw'
                    or bool(saved['normalized'].item()) or not bool(saved['has_spindle'].item())):
                raise ValueError(f'{file}: expected unnormalized v2_raw spindle data')
            if ids is None:
                ids = saved['neuron_ids']
                if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
                    raise ValueError(f'{file}: invalid neuron IDs')
            elif not np.array_equal(ids, saved['neuron_ids']):
                raise ValueError(f'{file}: inconsistent neuron IDs')
            paw = np.asarray(saved['paw'], dtype=np.float64)
            spikes = np.asarray(saved['spike_counts'], dtype=np.float64)
            x = np.asarray(saved['spindle'], dtype=np.float64)
        if (paw.ndim != 2 or paw.shape[1] != 2 or spikes.ndim != 2
                or paw.shape[0] != spikes.shape[0] or len(spikes) < 2
                or spikes.shape[1] != len(ids)
                or not np.isfinite(paw).all() or not np.isfinite(spikes).all()
                or np.any(spikes < 0) or not np.allclose(spikes, np.round(spikes))):
            raise ValueError(f'{path}: invalid paw/spike counts in {name}')
        if (x.shape != (len(spikes), 122) or not np.isfinite(x).all() or np.any(x < 0)):
            raise ValueError(f'{path}: invalid spindle channels in {name}')
        halves[name] = {'paw': paw, 'spikes': spikes, 'x': x}
    return ids, halves


def full_blocks(halves, dt=0.01, seconds=10):
    size = round(seconds / dt)
    if size < 2 or not np.isclose(size * dt, seconds):
        raise ValueError('Invalid block duration')
    blocks, audit = [], {}
    for name, half in halves.items():
        length = len(half['paw'])
        n, tail = divmod(length, size)
        if not n:
            raise ValueError(f'{name}: no complete movement block')
        blocks.extend((name, i * size, (i + 1) * size) for i in range(n))
        audit[name] = {'n_blocks': n, 'tail_bins_dropped': tail,
                       'samples': length, 'used_samples': n * size}
    if not all(any(h.startswith(phase) for h in halves) for phase in PHASES):
        raise ValueError('Both phases required')
    return blocks, audit
