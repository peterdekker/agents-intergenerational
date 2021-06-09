import numpy as np
from collections import defaultdict, Counter


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
                # Find, possibly multiple, most common elements
                most_common_list = Counter(affixes).most_common()
                max_freq = max([v for k, v in most_common_list])
                most_common = [k for k, v in most_common_list if v == max_freq]
                # Cell is filled if "" is not most common, or even among the most common
                if "" not in most_common:
                    filled += 1
                # most_common = Counter(affixes).most_common(1)[0][0]
                # if most_common != "":
                #     filled += 1

    return filled/total


def internal_filled(agents, aff_pos):
    filled_props = [agent_proportion_filled_cells(a, aff_pos) for a in agents]
    return np.mean(filled_props) if len(filled_props) > 0 else 0


def agent_affix_frequencies(agent, aff_pos, freq_dict):
    for lex_concept in agent.lex_concepts_type[f"{aff_pos}ing"]:
        for person in agent.persons:
            affix_list = agent.affixes[(lex_concept, person, aff_pos)]
            for aff in affix_list:
                freq_dict[f"'{aff}'-{person}"] += 1


def internal_affix_frequencies(agents, aff_pos):
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


def update_communicated_model_stats(model, prefix, suffix, prefixing, suffixing, l2):
    if prefixing:
        if l2:
            model.communicated_prefix_l2.append(prefix)
        else:
            model.communicated_prefix_l1.append(prefix)

    if suffixing:
        if l2:
            model.communicated_suffix_l2.append(suffix)
        else:
            model.communicated_suffix_l1.append(suffix)


def calculate_proportion_communicated(communicated_list):
    # Calculate proportion non-empty communications
    n_non_empty = len([s for s in communicated_list if s != ""])
    n_total = len(communicated_list)
    return n_non_empty/n_total if n_total > 0 else 0
