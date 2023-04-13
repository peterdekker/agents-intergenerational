
import numpy as np
from collections import defaultdict, Counter


def prop_internal_n_affixes(agent, affix_type):
    n_affixes_cells = []
    for lex_concept in agent.lex_concepts_type[affix_type]:
        for person in agent.persons:
            affixes = agent.affixes[(lex_concept, person, affix_type)]
            n_affixes = len(affixes)
            n_affixes_cells.append(n_affixes)

    return np.mean(n_affixes_cells) if len(n_affixes_cells) > 0 else None


def prop_internal_len(agent, affix_type):
    lens_affixes_cells = []
    for lex_concept in agent.lex_concepts_type[affix_type]:
        for person in agent.persons:
            affixes = agent.affixes[(lex_concept, person, affix_type)]
            lens_affixes = [len(a) for a in affixes]
            lens_affixes_cells += lens_affixes

    return np.mean(lens_affixes_cells) if len(lens_affixes_cells) > 0 else None


def prop_internal_nonzero(agent, affix_type):
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

# def prop_internal_n_affixes_agents(agents, affix_type):
#     n_affixes = [prop_internal_n_affixes(a, affix_type) for a in agents]
#     n_affixes = [x for x in n_affixes if x != None]
#     return np.mean(n_affixes) if len(n_affixes) > 0 else None

# def prop_internal_nonzero_agents(agents, affix_type):
#     filled_props = [prop_internal_nonzero(a, affix_type) for a in agents]
#     filled_props = [x for x in filled_props if x != None]
#     return np.mean(filled_props) if len(filled_props) > 0 else None


def apply_stat_agents(func, agents, affix_type):
    stat_agents = [func(a, affix_type) for a in agents]
    stat_agents = [x for x in stat_agents if x != None]
    return np.mean(stat_agents) if len(stat_agents) > 0 else None


def calculate_internal_stats(agents, generation, stats_entries):
    agents_l1 = [a for a in agents if not a.is_l2()]
    agents_l2 = [a for a in agents if a.is_l2()]

    for agents_set, agent_type in zip([agents_l1, agents_l2, agents], ["l1", "l2", "total"]):
        for affix_type in ["prefix", "suffix"]:
            # stat_nonzero = apply_stat_agents(prop_internal_nonzero, agents_set, affix_type)
            # stats_entry_nonzero = {"generation": generation, "stat_name": f"prop_internal_{affix_type}" if agent_type ==
            #                        "total" else f"prop_internal_{affix_type}_{agent_type}", "stat_value": stat_nonzero} # "proportion_l2": proportion_l2,
            # stats_entries.append(stats_entry_nonzero)

            # stat_len = apply_stat_agents(prop_internal_len, agents_set, affix_type)
            # stats_entry_len = {"generation": generation, "stat_name": f"prop_internal_len_{affix_type}" if agent_type ==
            #                        "total" else f"prop_internal_len_{affix_type}_{agent_type}", "stat_value": stat_len} # "proportion_l2": proportion_l2,
            # stats_entries.append(stats_entry_len)

            # stat_n_affixes = apply_stat_agents(prop_internal_n_affixes, agents_set, affix_type)
            # stats_entry_n_affixes = {"generation": generation, "stat_name": f"prop_internal_n_affixes_{affix_type}" if agent_type ==
            #                          "total" else f"prop_internal_n_affixes_{affix_type}_{agent_type}", "stat_value": stat_n_affixes} #"proportion_l2": proportion_l2,
            # stats_entries.append(stats_entry_n_affixes)

            for stat_name, stat_func in [("prop_internal", prop_internal_nonzero), ("prop_internal_len", prop_internal_len), ("prop_internal_n_affixes", prop_internal_n_affixes)]:
                stat_value = apply_stat_agents(stat_func, agents_set, affix_type)
                stats_entry = {"generation": generation, "stat_name": f"{stat_name}_{affix_type}" if agent_type ==
                               "total" else f"{stat_name}_{affix_type}_{agent_type}", "stat_value": stat_value}
                stats_entries.append(stats_entry)


def calculate_correct_interactions(correct_interactions, total_interactions, current_generation, stats_entries):
    # Proportion correct interactions is calculated based on total_interactions:
    # number of interactions where initiator actually speaks.
    # Interactions where initiator does not speak, because system is empty, are excluded
    proportion_correct_interactions = correct_interactions / total_interactions if total_interactions > 0 else 0
    stats_entry_prop_correct = {"generation": current_generation,  # "proportion_l2": proportion_l2,
                                "stat_name": "prop_correct", "stat_value": proportion_correct_interactions}
    stats_entries.append(stats_entry_prop_correct)


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
