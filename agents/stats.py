
import numpy as np
import os
from collections import defaultdict, Counter
import pandas as pd


def prop_internal_n_unique(agent, affix_type):
    all_affixes = []
    for lex_concept in agent.lex_concepts_type[affix_type]:
        for person in agent.persons:
            affixes = agent.affixes[(lex_concept, person, affix_type)]
            all_affixes += affixes
    n_affixes_unique = len(set(all_affixes))

    return n_affixes_unique if n_affixes_unique > 0 else None


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


def affix_sample_diagnosis(agents, output_dir, interactions_per_generation, proportion_l2, run_id):
    agents_prev_gen_l2 = [a for a in agents if a.is_l2()]
    np.random.shuffle(agents_prev_gen_l2)
    sample_agent = agents_prev_gen_l2[0]
    affixes_emptysymbol = {}
    for key in sample_agent.affixes:
        affixes_emptysymbol[key] = ["∅" if aff == "" else aff for aff in sample_agent.affixes[key]]
    affix_sample_df = pd.DataFrame.from_dict(affixes_emptysymbol, orient="index")
    affix_sample_df.to_csv(os.path.join(
        output_dir, f"affix_sample_ipg{interactions_per_generation}_prop{proportion_l2}_run{run_id}.csv"))


def apply_stat_agents(func, agents, affix_type):
    stat_agents = [func(a, affix_type) for a in agents]
    stat_agents = [x for x in stat_agents if x != None]
    return np.mean(stat_agents) if len(stat_agents) > 0 else None


def calculate_internal_stats(agents, generation, correct_interactions, total_interactions, stats_entries):
    agents_l1 = [a for a in agents if not a.is_l2()]
    agents_l2 = [a for a in agents if a.is_l2()]

    for agents_set, agent_type in zip([agents_l1, agents_l2, agents], ["l1", "l2", "total"]):
        for affix_type in ["prefix", "suffix"]:

            # Create dictionary with key-value pair per agent-based statistic
            stats_entry = {"generation": generation}
            for stat_name, stat_func in [("prop_internal", prop_internal_nonzero), ("prop_internal_len", prop_internal_len), ("prop_internal_n_affixes", prop_internal_n_affixes), ("prop_internal_n_unique", prop_internal_n_unique)]:
                stat_name_key = f"{stat_name}_{affix_type}" if agent_type == "total" else f"{stat_name}_{affix_type}_{agent_type}"
                stat_value = apply_stat_agents(stat_func, agents_set, affix_type)
                stats_entry[stat_name_key] = stat_value

            # Add population-wide prop correct
            stats_entry["prop_correct"] = correct_interactions / \
                total_interactions if total_interactions > 0 else 0

            # Append this dictionary (one row for future dataframe) to stats_entries
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
    n_non_empty = len([s for s in communicated_list if s != ""])
    n_total = len(communicated_list)
    # Clear communicated list, so new proportion is calculated for next COMMUNICATED_STATS_AFTER_GENERATIONS generations
    communicated_list.clear()
    # Calculate proportion
    return n_non_empty/n_total if n_total > 0 else None
