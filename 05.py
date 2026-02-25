"""
Yuli Tshuva
Building upon gemini code for Optimal Transport.
The coupling matrix T represents how much "mass" from each node in G1 is transported to each node in G2.
The sum of each row in T corresponds to the weight of the node in G1.
"""

import numpy as np
import networkx as nx
import ot
from scipy.spatial.distance import cdist
from scipy.linalg import expm


class GraphSimilarityModel:
    def __init__(self, method='heat_kernel', alpha=0.5, t=1.0):
        self.method = method
        self.alpha = alpha
        self.t = t

        if method not in ['shortest_path', 'heat_kernel']:
            raise ValueError("Method must be 'shortest_path' or 'heat_kernel'.")

    def compute_structure_matrix(self, G, weight_attr='weight'):
        # Find the nodes
        nodes = list(G.nodes)
        n = len(nodes)

        if self.method == 'shortest_path':
            # Create a temporary distance attribute: High Importance = Low Distance
            for u, v, d in G.edges(data=True):
                importance = d.get(weight_attr, 1.0)
                G[u][v]['_tmp_dist'] = 1.0 / importance if importance > 0 else 1e6

            # Compute Dijkstra lengths
            path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='_tmp_dist'))
            C = np.zeros((n, n))
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    # Use a large constant for unreachable nodes
                    C[i, j] = path_lengths[u].get(v, 10.0)

        elif self.method == 'adjacency':
            # Use raw weights. High importance = Low cost in C matrix.
            A = nx.to_numpy_array(G, nodelist=nodes, weight=weight_attr)
            C = A.max() - A
            np.fill_diagonal(C, 0)

        elif self.method == 'heat_kernel':
            # Weighted Laplacian automatically treats 'weight' as conductance (flow)
            L = nx.laplacian_matrix(G, nodelist=nodes, weight=weight_attr).toarray()
            H = expm(-self.t * L)
            C = -np.log(H + 1e-9)

        # Normalize the structural matrix so it's comparable to feature distances
        if C.max() > 0: C /= C.max()
        return C

    def compare(self, G1, feat1, G2, feat2):
        C1 = self.compute_structure_matrix(G1)
        C2 = self.compute_structure_matrix(G2)

        # Feature cost (Euclidean distance between vectors)
        M = cdist(feat1, feat2, metric='sqeuclidean')
        if M.max() > 0: M /= M.max()

        # Mass distributions (Uniform across nodes)
        p = ot.unif(len(G1.nodes))
        q = ot.unif(len(G2.nodes))

        # Fused Gromov-Wasserstein
        dist, log = ot.gromov.fused_gromov_wasserstein2(
            M, C1, C2, p, q, alpha=self.alpha, log=True
        )
        return dist, log['T']


# --- Execution ---

# 1. Setup Graph 1: Path with an "Important" middle edge
G1 = nx.path_graph(4)
features1 = np.random.rand(4, 5)
for u, v in G1.edges():
    G1[u][v]['weight'] = 2.0 if (u == 1 and v == 2) else 1.0

# 2. Setup Graph 2: Star with "Important" first two edges
G2 = nx.star_graph(5)
features2 = np.random.rand(6, 5)
for i, (u, v) in enumerate(G2.edges()):
    G2[u][v]['weight'] = 2.0 if i < 2 else 1.0

# 3. Initialize and Run
model = GraphSimilarityModel(method='heat_kernel', alpha=0.5)
distance, coupling = model.compare(G1, features1, G2, features2)

print(f"Similarity Distance: {distance:.4f}")