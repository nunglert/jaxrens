"""Hand-built layered structures shared by the layered-geometry tests."""

import numpy as np

C, LI = 0, 1
A1 = np.array([2.46, 0.0, 0.0])
A2 = np.array([1.23, 2.46 * np.sqrt(3) / 2, 0.0])
HOLLOW = 2.0 * (A1 + A2) / 3.0


def graphite(spacings, li_per_gallery=(), z0=0.0, reps=3):
    """AA-stacked graphite, ``reps`` x ``reps`` in plane.

    Args:
        spacings: Gallery heights (Angstrom), one per sheet.
        li_per_gallery: Number of Li at hollow sites in mid-gallery, per
            gallery (missing entries mean none).
        z0: Height of sheet 0.

    Returns:
        ``(positions, types, cell)`` as numpy arrays; C atoms come first,
        sheet by sheet.
    """
    a, b = reps * A1, reps * A2
    cell = np.array([a, b, [0.0, 0.0, float(sum(spacings))]])
    base = [i * A1 + j * A2 for i in range(reps) for j in range(reps)]
    tops = np.concatenate(
        [np.cumsum([0.0, *spacings[:-1]]) + z0, [z0 + sum(spacings)]]
    )
    pos, typ = [], []
    for l in range(len(spacings)):
        for p in base:
            for d in (np.zeros(3), (A1 + A2) / 3.0):
                pos.append(p + d + [0.0, 0.0, tops[l]])
                typ.append(C)
    for g, n_li in enumerate(li_per_gallery):
        mid = 0.5 * (tops[g] + tops[g + 1])
        for p in base[:n_li]:
            pos.append(p + HOLLOW + [0.0, 0.0, mid])
            typ.append(LI)
    return np.array(pos), np.array(typ, dtype=np.int32), cell
