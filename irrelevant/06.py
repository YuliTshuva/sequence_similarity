"""
Yuli Tshuva
Building upon gemini code for Optimal Transport.
The coupling matrix T represents how much "mass" from each node in G1 is transported to each node in G2.
The sum of each row in T corresponds to the weight of the node in G1.
"""

# Import
import numpy as np
import networkx as nx
import ot
from scipy.spatial.distance import cdist

# Constant
FEATURES_DISTANCE_METRIC = 'sqeuclidean'  # Can be 'euclidean' as well

class GraphSimilarityModel:
    def __init__(self, alpha=0.5):
        # Convert to a weight for the Gromov-Wasserstein cost
        self.alpha = 1 / (1 + alpha)

    def compute_structure_matrix(self, G, weight_attr='weight'):
        # Find the nodes
        nodes = list(G.nodes)
        n = len(nodes)



    def compare(self, G1, feat1, G2, feat2):
        C1 = self.compute_structure_matrix(G1)
        C2 = self.compute_structure_matrix(G2)

        # Feature cost (Euclidean distance between vectors)
        M = cdist(feat1, feat2, metric=FEATURES_DISTANCE_METRIC)
        if M.max() > 0:
            M /= M.max()
        else:
            raise Warning("Feature distance matrix is zero. Check your features or distance metric.")

        # Mass distributions is Uniform across nodes
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
model = GraphSimilarityModel(alpha=1)
distance, coupling = model.compare(G1, features1, G2, features2)

print(f"Similarity Distance: {distance:.4f}")