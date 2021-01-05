import numpy as np
from itertools import combinations
from distance import jaccard


def morph_complexity(agent):
    # TODO: optimize, get rid of loops
    lengths = []
    for lex_concept in agent.lex_concepts:
        for person in agent.persons:
            for affix_position in ["prefix", "suffix"]:
                # Length is also calculated for empty affixes list (in L2 agents)
                n_affixes = len(set(agent.affixes[(lex_concept, person, affix_position)]))
                lengths.append(n_affixes)
    mean_length = np.mean(lengths)  # if len(lengths)>0 else 0
    return mean_length


def compute_colour(agent):
    agg = morph_complexity(agent) * 100
    return [250, 80, agg]


def compute_global_dist(agents, lex_concepts, persons):
    dists = []
    # Compute test statistic by sampling some pairs
    #agents_sample = np.random.choice(agents, SAMPLE, replace=False)
    # agents1 = agents#agents_sample[:HSAMPLE]
    # agents2 = agents#agents_sample[HSAMPLE:]
    for agent1, agent2 in combinations(agents, 2):
        for lex_concept in lex_concepts:
            for person in persons:
                for affix_position in ["prefix", "suffix"]:
                    aff1 = agent1.affixes[(lex_concept, person, affix_position)]
                    aff2 = agent2.affixes[(lex_concept, person, affix_position)]
                    if aff1 == [] and aff2 == []:
                        # jaccard([],[]) is undefined, these comparisons will be skipped (=undefined)
                        continue
                    jaccard_dist = jaccard(aff1, aff2)
                    dists.append(jaccard_dist)
    global_model_distance = np.mean(dists)
    return global_model_distance
