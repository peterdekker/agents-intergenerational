from constants import SAMPLE, HSAMPLE
import numpy as np
from scipy.stats import entropy
import time

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
            agg_agent1 = compute_agent_agg(agent1.affixes)
            agg_agent2 = compute_agent_agg(agent2.affixes)
            dist = np.abs(agg_agent1-agg_agent2)
            cumul_model_distance += dist
            n_pairs += 1
    global_model_distance = float(cumul_model_distance)/float(n_pairs)
    return global_model_distance


def compute_agent_agg(affixes):
    # TODO: optimize, get rid of loops
    entropies = []
    #start = time.time()
    #affixes_sample = np.random.choice(list(affixes.keys()), SAMPLE, replace=False)
    for lex_concept,persons_dict in affixes.items():
        for person, affixes_dict in persons_dict.items():
            for affix, prob_dict in affixes_dict.items():
                ent = len(prob_dict) #entropy(values, base=2)
                entropies.append(ent)
    mean_entropy = np.mean(entropies)
    #end=time.time()
    #print(f"colour:{end-start}")
    return mean_entropy

def compute_agent_colour(affixes):
    color_scale = 255
    agg = compute_agent_agg(affixes)
    return [agg*color_scale, 100, 100]