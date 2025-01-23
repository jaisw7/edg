# -*- coding: utf-8 -*-

"""Mappings between the internal ordering of nodes, and those of external formats

"""
import numpy as np


class GmshNodeMaps(object):
    """Mappings between the internal node ordering and that of Gmsh"""

    to_internal = {
        ("tri", 3): np.array([0, 1, 2]),
        ("tri", 6): np.array([0, 2, 5, 1, 4, 3]),
        ("tri", 10): np.array([0, 3, 9, 1, 2, 6, 8, 7, 4, 5]),
        ("tri", 15): np.array([0, 4, 14, 1, 2, 3, 8, 11, 13, 12, 9, 5, 6, 7, 10]),
        ("tri", 21): np.array(
            [0, 5, 20, 1, 2, 3, 4, 10, 14, 17, 19, 18, 15, 11, 6, 7, 8, 9, 13, 16, 12]
        ),
        ("line", 2): np.array([0, 1]),
    }

    from_internal = {k: np.argsort(v) for k, v in to_internal.items()}
