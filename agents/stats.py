import numpy as np
from itertools import combinations
from collections import defaultdict, Counter
from distance import jaccard


def agent_proportion_filled_cells(agent, aff_pos):
    total = 0
    filled = 0
    for lex_concept in agent.lex_concepts_type[f"{aff_pos}ing"]:
        for person in agent.persons:
            total += 1
            # affix_set = set(agent.affixes[(lex_concept, person, aff_pos)])
            # affix_set.discard("")
            # if len(affix_set) > 0:
            #     filled += 1
            affixes = agent.affixes[(lex_concept, person, aff_pos)]
            if len(affixes) > 0:
                most_common = Counter(affixes).most_common(1)[0][0]
                if most_common != "":
                    filled += 1
    return filled/total


def global_filled(agents, aff_pos):
    filled_props = [agent_proportion_filled_cells(a, aff_pos) for a in agents]
    return np.mean(filled_props) if len(filled_props) > 0 else 0


def agent_affix_frequencies(agent, aff_pos, freq_dict):
    for lex_concept in agent.lex_concepts_type[f"{aff_pos}ing"]:
        for person in agent.persons:
            affix_list = agent.affixes[(lex_concept, person, aff_pos)]
            for aff in affix_list:
                freq_dict[f"'{aff}'-{person}"] += 1


def global_affix_frequencies(agents, aff_pos):
    freq_dict = defaultdict(int)
    for a in agents:
        agent_affix_frequencies(a, aff_pos, freq_dict)
    return freq_dict


def compute_colour(agent):
    agg_prefix = agent_proportion_filled_cells(agent, "prefix") * 50
    agg_suffix = agent_proportion_filled_cells(agent, "suffix") * 50
    # HSL: H->0-360,  S->0-100%, L->100% L50% is maximum color, 100% is white
    return {"prefix": colour_str([250, 80, agg_prefix]), "suffix": colour_str([250, 80, agg_suffix])}


def colour_str(c):
    return f"hsl({c[0]},{c[1]}%,{c[2]}%)"


def global_dist(agents, lex_concepts, persons):
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
    return np.mean(dists) if len(dists) > 0 else 0
