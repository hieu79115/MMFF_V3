from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class GraphSpec:
    num_node: int
    self_links: List[Tuple[int, int]]
    inward: List[Tuple[int, int]]


def _to_zero_based(edge_list_1_based: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(i - 1, j - 1) for (i, j) in edge_list_1_based]


def ntu_rgbd_25() -> GraphSpec:
    """NTU RGB+D 25-joint graph used by ST-GCN.

    The joint indexing follows the common NTU 25-joint convention used in ST-GCN repos.
    """

    # Self links
    self_links = [(i, i) for i in range(25)]

    # 1-based inward edges from ST-GCN reference implementation (commonly used)
    inward_1b = [
        (1, 2),
        (2, 21),
        (3, 21),
        (4, 3),
        (5, 21),
        (6, 5),
        (7, 6),
        (8, 7),
        (9, 21),
        (10, 9),
        (11, 10),
        (12, 11),
        (13, 1),
        (14, 13),
        (15, 14),
        (16, 15),
        (17, 1),
        (18, 17),
        (19, 18),
        (20, 19),
        (22, 23),
        (23, 8),
        (24, 25),
        (25, 12),
    ]
    inward = _to_zero_based(inward_1b)

    return GraphSpec(num_node=25, self_links=self_links, inward=inward)


def ut_mhad_20() -> GraphSpec:
    """A reasonable 20-joint human body graph for UT-MHAD.

    UT-MHAD skeleton has 20 joints. Public codebases differ slightly on exact edges.
    This definition matches a typical Kinect-like 20-joint tree and is good as a starting point.
    """

    self_links = [(i, i) for i in range(20)]

    # 1-based edges for a Kinect-style 20 joint topology.
    inward_1b = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (3, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (3, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (1, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (1, 18),
        (18, 19),
        (19, 20),
    ]
    inward = _to_zero_based(inward_1b)

    return GraphSpec(num_node=20, self_links=self_links, inward=inward)


def _edge2mat(edges: List[Tuple[int, int]], num_node: int) -> np.ndarray:
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edges:
        A[j, i] = 1.0
    return A


def _normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = A.sum(0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node), dtype=np.float32)
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return (A @ Dn).astype(np.float32)


def get_adjacency_matrix(dataset: str) -> np.ndarray:
    """Return ST-GCN partitioned adjacency matrix A with shape (K, V, V).

    K=3 partitions (root/self, inward, outward) is the common ST-GCN setup.
    """

    ds = dataset.lower()
    if ds in {"ntu", "ntu60", "ntu_rgbd", "ntu-rgbd", "ntu120"}:
        spec = ntu_rgbd_25()
    elif ds in {"ut", "utmhad", "ut-mhad", "ut_mhad", "utd", "utd-mhad"}:
        spec = ut_mhad_20()
    else:
        raise ValueError(f"Unknown dataset graph '{dataset}'. Use 'ntu60' or 'ut_mhad'.")

    inward = spec.inward
    outward = [(j, i) for (i, j) in inward]

    A0 = _edge2mat(spec.self_links, spec.num_node)
    A1 = _edge2mat(inward, spec.num_node)
    A2 = _edge2mat(outward, spec.num_node)

    A = np.stack([
        _normalize_digraph(A0),
        _normalize_digraph(A1),
        _normalize_digraph(A2),
    ], axis=0)
    return A
