from constants import SAMPLE, HSAMPLE
import numpy as np
from scipy.stats import entropy


def compute_global_dist(agents):
    cumul_model_distance = 0
    n_pairs = 0
    # Compute test statistic by sampling some pairs
    agents_sample = np.random.choice(agents, SAMPLE, replace=False)
    agents1 = agents_sample[:HSAMPLE]
    agents2 = agents_sample[HSAMPLE:]
    for agent1 in agents1:
        for agent2 in agents2:
            # Euclidean distance
            dist = np.linalg.norm(agent1.language - agent2.language)
            cumul_model_distance += dist
            n_pairs += 1
    global_model_distance = float(cumul_model_distance)/float(n_pairs)
    return global_model_distance


def compute_language_agg(affixes):
    # Only look at first three concepts: every concept will be a channel
    # Scale by total possible sum
    color_scale = 255
    entropies = []
    print(affixes)
    for lex_concept in affixes:
        for person in affixes[lex_concept]:
            for affix in affixes[lex_concept][person]:
                values = affixes[lex_concept][person].values()
                print(values)
                entropies.append(entropy(values))
    mean_entropy = np.mean(entropies)
    return [mean_entropy*color_scale, 100, 100]
