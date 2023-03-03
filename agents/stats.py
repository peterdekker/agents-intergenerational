
import numpy as np
from collections import defaultdict, Counter


def prop_internal_filled(agent, affix_type):
    prop_filled_cells = []
    for lex_concept in agent.lex_concepts_type[affix_type]:
        for person in agent.persons:
            affixes = agent.affixes[(lex_concept, person, affix_type)]
            n_affixes = len(affixes)
            n_nonzero = len([x for x in affixes if x != ""])
            if n_affixes > 0:
                prop_filled_cell = n_nonzero/n_affixes
                prop_filled_cells.append(prop_filled_cell)

    return np.mean(prop_filled_cells) if len(prop_filled_cells) > 0 else None


def prop_internal_filled_agents(agents, affix_type):
    filled_props = [prop_internal_filled(a, affix_type) for a in agents]
    filled_props = [x for x in filled_props if x != None]
    return np.mean(filled_props) if len(filled_props) > 0 else None


def calculate_internal_stats(agents, generation, proportion_l2, stats_entries):
    agents_l1 = [a for a in agents if not a.is_l2()]
    agents_l2 = [a for a in agents if a.is_l2()]

    for agents_set, agent_type in zip([agents_l1, agents_l2, agents], ["l1", "l2", "total"]):
        for affix_type in ["prefix", "suffix"]:
            stat = prop_internal_filled_agents(agents_set, affix_type)
            stats_entry = {"generation": generation, "proportion_l2": proportion_l2, "stat_name": f"prop_internal_{affix_type}" if agent_type == "total" else f"prop_internal_{affix_type}_{agent_type}", "stat_value": stat}
            stats_entries.append(stats_entry)


def update_communicated_model_stats(model, prefix, suffix, prefixing, suffixing, l2):
    if prefixing:
        assert isinstance(prefix, str)  # check that it is not none
        model.communicated_prefix.append(prefix)
        if l2:
            model.communicated_prefix_l2.append(prefix)
        else:
            model.communicated_prefix_l1.append(prefix)

    if suffixing:
        assert isinstance(suffix, str)  # check that it is not none
        model.communicated_suffix.append(suffix)
        if l2:
            model.communicated_suffix_l2.append(suffix)
        else:
            model.communicated_suffix_l1.append(suffix)


def prop_communicated(communicated_list, label=""):
    # if label=="Suffix L2":
    #     print(communicated_list)
    # Calculate proportion non-empty communications
    n_non_empty = len([s for s in communicated_list if s != ""])
    n_total = len(communicated_list)
    # if n_total == 0 and "L1" in label:
    #     print(f"{label}:empty")
    # Clear communicated list, so new proportion is calculated for next COMMUNICATED_STATS_AFTER_GENERATIONS generations
    communicated_list.clear()
    # Calculate proportion
    return n_non_empty/n_total if n_total > 0 else None
