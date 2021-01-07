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

# TODO: possibly parametrize for affix position
def proportion_filled_entries(agent):
    total = {"prefix": 0, "suffix": 0}
    filled = {"prefix": 0, "suffix": 0}
    for lex_concept in agent.lex_concepts:
        for person in agent.persons:
            for aff_pos in ["prefix", "suffix"]:
                if agent.lex_concept_data[lex_concept][f"{aff_pos}ing"]:
                    total[aff_pos] +=1
                    affix_set = set(agent.affixes[(lex_concept, person, aff_pos)])
                    affix_set.discard("")
                    if len(affix_set) > 0:
                        filled[aff_pos]+=1

    proportion_prefix = filled["prefix"]/total["prefix"]
    proportion_suffix = filled["suffix"]/total["suffix"]
    return {"prefix": proportion_prefix, "suffix": proportion_suffix}


def compute_colour(agent):
    agg = proportion_filled_entries(agent)
    agg_prefix = agg["prefix"] * 50
    agg_suffix = agg["suffix"] * 50
    #HSL: H->0-360,  S->0-100%, L->100% L50% is maximum color, 100% is white
    return {"prefix": colour_str([250, 80, agg_prefix]), "suffix": colour_str([250, 80, agg_suffix])}

def colour_str(c):
    return f"hsl({c[0]},{c[1]}%,{c[2]}%)"

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
