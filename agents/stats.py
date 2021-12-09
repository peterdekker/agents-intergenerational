
import numpy as np
from collections import defaultdict, Counter


def prop_internal_filled(agent, aff_pos):
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


def prop_internal_filled_agents(agents, aff_pos):
    filled_props = [prop_internal_filled(a, aff_pos) for a in agents]
    return np.mean(filled_props) if len(filled_props) > 0 else 0


def internal_affix_frequencies(agent, aff_pos, freq_dict):
    for lex_concept in agent.lex_concepts_type[f"{aff_pos}ing"]:
        for person in agent.persons:
            affix_list = agent.affixes[(lex_concept, person, aff_pos)]
            for aff in affix_list:
                freq_dict[f"'{aff}'-{person}"] += 1


def internal_affix_frequencies_agents(agents, aff_pos):
    freq_dict = defaultdict(int)
    for a in agents:
        internal_affix_frequencies(a, aff_pos, freq_dict)
    return freq_dict


def compute_colours_agents(agents):
    for agent in agents:
        agent.colours = compute_colours(agent)


def compute_colours(agent):
    agg_prefix = prop_internal_filled(agent, "prefix") * 50
    agg_suffix = prop_internal_filled(agent, "suffix") * 50
    # HSL: H->0-360,  S->0-100%, L->100% L50% is maximum color, 100% is white
    return {"prefix": colour_str([250, 80, agg_prefix]), "suffix": colour_str([250, 80, agg_suffix])}


def colour_str(c):
    return f"hsl({c[0]},{c[1]}%,{c[2]}%)"


def update_communicated_model_stats(model, prefix, suffix, prefixing, suffixing, l2, step):
    if prefixing:
        if l2:
            model.communicated_prefix_l2[-1].append(prefix)
        else:
            model.communicated_prefix_l1[-1].append(prefix)

    if suffixing:
        if l2:
            model.communicated_suffix_l2[-1].append(suffix)
        else:
            model.communicated_suffix_l1[-1].append(suffix)


def prop_communicated(communicated_list):
    last_comm_list = communicated_list[-1]
    # Calculate proportion non-empty communications
    n_non_empty = len([s for s in last_comm_list if s != ""])
    n_total = len(last_comm_list)
    # Calculate proportion
    return n_non_empty/n_total if n_total > 0 else 0
