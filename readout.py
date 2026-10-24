"""Full-block, native-coordinate position tuning; based on position_readout."""

import numpy as np

AXES = {'phase1': 0, 'phase3': 1}  # flattened paw columns 9 and 11
PAW_COLUMNS = {'phase1': 9, 'phase3': 11}


def fit_ranges(halves, blocks, bins):
    if bins < 4:
        raise ValueError('At least four position bins required')
    ranges = {}
    for phase, col in AXES.items():
        samples = np.concatenate([halves[h]['paw'][a:b, col]
                                  for h, a, b in blocks if h.startswith(phase)])
        lo, hi = np.percentile(samples, [1, 99])
        if not np.isfinite(lo + hi) or hi <= lo:
            raise ValueError(f'{phase}: collapsed position range')
        ranges[phase] = [float(lo), float(hi)]
    return ranges


def readout(halves, blocks, ranges, output, bins, dt=0.01,
            min_seconds=1.0, min_blocks=3):
    """Exclude out-of-P1/P99 samples; missing gated bins stay NaN, not zero.

    output(name, start, stop) returns spike counts per sample (real or virtual).
    Virtual deterministic ReLU rates are multiplied by dt before this call.
    """
    curves, audit = {}, {}
    for phase, col in AXES.items():
        lo, hi = ranges[phase]
        occupancy, counts = [], []
        outside = total = 0
        for h, a, b in blocks:
            if not h.startswith(phase):
                continue
            pos = halves[h]['paw'][a:b, col]
            y = np.asarray(output(h, a, b))
            if (y.ndim != 2 or y.shape[0] != len(pos) or not np.isfinite(y).all()
                    or np.any(y < 0)):
                raise ValueError('Invalid output block')
            u = (pos - lo) / (hi - lo)
            valid = np.isfinite(u) & (u >= 0) & (u <= 1)
            outside += int((~valid).sum())
            total += len(pos)
            k = np.minimum((u[valid] * bins).astype(int), bins - 1)
            occupancy.append(np.bincount(k, minlength=bins))
            sums = np.zeros((bins, y.shape[1]))
            np.add.at(sums, k, y[valid])
            counts.append(sums)
        if not occupancy:
            raise ValueError(f'{phase}: no complete blocks')
        occ = np.stack(occupancy)
        duration = occ.sum(axis=0) * dt
        independent = (occ > 0).sum(axis=0)
        rate = np.divide(np.stack(counts).sum(axis=0), duration[:, None],
                         out=np.full((bins, counts[0].shape[1]), np.nan),
                         where=duration[:, None] > 0)
        rate[(duration < min_seconds) | (independent < min_blocks)] = np.nan
        curves[phase] = rate
        audit[phase] = {'outside': outside, 'total': total, 'column': PAW_COLUMNS[phase],
                        'dwell_seconds': duration.tolist(),
                        'independent_blocks': independent.tolist()}
    return curves, audit
