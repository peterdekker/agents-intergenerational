from constants import SAMPLE, HSAMPLE
import numpy as np
from scipy.spatial.distance import euclidean
from itertools import combinations
import time

def compute_global_dist(agents):
    cumul_model_distance = 0
    n_pairs = 0
    # Compute test statistic by sampling some pairs
    #agents_sample = np.random.choice(agents, SAMPLE, replace=False)
    #agents1 = agents#agents_sample[:HSAMPLE]
    #agents2 = agents#agents_sample[HSAMPLE:]
    for agent1, agent2 in combinations(agents, 2):
        # Euclidean distance
        _, agg_agent1 = agent1.compute_agg()
        _, agg_agent2 = agent2.compute_agg()
        dist = euclidean(agg_agent1, agg_agent2)
        cumul_model_distance += dist
        n_pairs += 1
    global_model_distance = float(cumul_model_distance)/float(n_pairs)
    return global_model_distance
