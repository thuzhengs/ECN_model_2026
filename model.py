"""Muscle-bound, fixed-connectivity ECN forward model (no GLM fitting)."""

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

N_MUSCLES = 61
N_CHANNELS = 122


def muscle_clusters(halves, blocks, n_clusters=4):
    """Cluster the sum of paired afferents on model inputs only, never paw data."""
    if not 2 <= n_clusters <= 12:
        raise ValueError('n_clusters must be in [2, 12]')
    x = np.concatenate([halves[h]['x'][a:b] for h, a, b in blocks])
    signals = x[:, :N_MUSCLES] + x[:, N_MUSCLES:]
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.nan_to_num(np.corrcoef(signals.T))
    distances = 1.0 - corr
    np.fill_diagonal(distances, 0.0)
    labels = hierarchy.fcluster(
        hierarchy.linkage(squareform(distances, checks=False), method='average'),
        n_clusters, criterion='maxclust') - 1
    effective = int(labels.max()) + 1
    means = np.full((effective, effective), np.inf)
    for i in range(effective):
        for j in range(effective):
            if i != j:
                means[i, j] = float(corr[np.ix_(labels == i, labels == j)].mean())
    pairing = means.argmin(axis=1)
    return labels, pairing, {'cluster_sizes': np.bincount(labels).tolist(),
                             'n_clusters_effective': effective,
                             'inter_cluster_corr': [[None if i == j else float(means[i, j])
                                                     for j in range(effective)]
                                                    for i in range(effective)]}


def weights(n, ni, n_cells, seed, labels, pairing):
    """Reproduce the structured-synergy/type-gain sampling of the full run.

    The original nominal-grid conditions use topology_structured's draw
    order; off-grid exact-count conditions use the run_matched_population
    extension's draw order. Both bind Ia/II for every chosen muscle.
    """
    if not (1 <= n <= N_MUSCLES and 0 <= ni <= n and n_cells >= 1):
        raise ValueError('Invalid N, NI, or population size')
    labels, pairing = np.asarray(labels), np.asarray(pairing)
    if (labels.shape != (N_MUSCLES,) or pairing.shape != (int(labels.max()) + 1,)
            or np.any(labels < 0) or np.any(labels >= len(pairing))
            or np.any(pairing < 0) or np.any(pairing >= len(pairing))):
        raise ValueError('Invalid cluster labels or pairing')
    widths = (1, 2, 4, 8, 16, 32, 61)
    fractions = (0, .125, .25, .5, .75, .875, 1)
    nominal = next((rho for rho in fractions if round(n * rho) == ni), None)
    on_grid = n in widths and nominal is not None
    rng = np.random.default_rng(seed)
    sizes = np.bincount(labels, minlength=len(pairing)).astype(float)
    w = np.zeros((N_CHANNELS, n_cells), dtype=np.float64)
    fills = []
    for cell in range(n_cells):
        e_cl = rng.choice(len(sizes), p=sizes / sizes.sum())
        i_cl = int(pairing[e_cl])
        e_pool = np.flatnonzero(labels == e_cl)
        i_pool = np.flatnonzero(labels == i_cl)
        n_e = n - ni
        e = list(rng.choice(e_pool, min(len(e_pool), n_e), replace=False))
        i = list(rng.choice(i_pool, min(len(i_pool), ni), replace=False))
        fills.append(n - len(e) - len(i))
        used = set(e) | set(i)
        for group, target in ((e, n_e), (i, ni)):
            if len(group) < target:
                remaining = [m for m in range(N_MUSCLES) if m not in used]
                added = list(rng.choice(remaining, target - len(group), replace=False))
                group.extend(added)
                used.update(added)
        for group, sign in ((e, 1), (i, -1)):
            if group:
                group = np.asarray(group, dtype=int)
                channels = np.r_[group, group + N_MUSCLES]
                w[channels, cell] = sign * rng.uniform(0, 1, len(channels)) / np.sqrt(n)
    type_rng = np.random.default_rng(seed + 9011)
    ratio = type_rng.lognormal(0.0, 1.0, n_cells)
    w[:N_MUSCLES] *= np.sqrt(ratio)[None, :]
    w[N_MUSCLES:] *= (1.0 / np.sqrt(ratio))[None, :]
    return w, {'within_cluster_frac_mean': float(1 - np.mean(fills) / n),
               'within_cluster_frac_min': float(1 - np.max(fills) / n),
               'nominal_representative': nominal if on_grid else None}


def relu_rate(x, w):
    """Deterministic max(spindle @ signed weights, 0), no fitted offset/scale."""
    return np.maximum(np.asarray(x) @ np.asarray(w), 0.0)
