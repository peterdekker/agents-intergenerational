# import editdistance
import math
import numpy as np
import os
from itertools import chain

from agents.config import logging, RG, CURRENTDIR, GENERALIZE_PERSONS, GENERALIZE_LEX_CONCEPTS, GENERALIZE_PREFIX_SUFFIX

from collections import Counter


from pyclts import CLTS
import requests
import shutil
import math


def download_if_needed(archive_path, archive_url, file_path, label):
    if not os.path.exists(file_path):
        with open(archive_path, 'wb') as f:
            print(f"Downloading {label} from {archive_url}")
            try:
                r = requests.get(archive_url, allow_redirects=True)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            # Write downloaded content to file
            f.write(r.content)
            if archive_path.endswith(".tar.gz"):
                print("Unpacking archive.")
                shutil.unpack_archive(archive_path, CURRENTDIR)


def load_clts(clts_path):
    return CLTS(clts_path)


def most_common(lst):
    most_common = []
    if len(lst) > 0:
        data = Counter(lst)
        most_common = [data.most_common(1)[0][0]]
    return most_common


def lookup_lex_concept(signal_form, lex_concepts, lex_concept_data):
    retrieved_lex_concept = None
    for lex_concept in lex_concepts:
        if lex_concept_data[lex_concept]["form"] == signal_form:
            retrieved_lex_concept = lex_concept
            break
    if not retrieved_lex_concept:
        raise Exception("Inferred lex concept cannot be None!")
    return retrieved_lex_concept


# , ambiguity):
def infer_person_from_signal(lex_concept, lex_concept_data, affixes, persons, signal):
    logging.debug("Person not given via context, inferring from affix.")

    possible_persons = []
    if lex_concept_data[lex_concept]["prefix"]:
        possible_persons += infer_possible_persons("prefix", signal.prefix, persons,
                                                   affixes, lex_concept)
    if lex_concept_data[lex_concept]["suffix"]:
        possible_persons += infer_possible_persons("suffix",
                                                   signal.suffix, persons,
                                                   affixes, lex_concept)

    # If no possible persons (because no affix, or empty internal suffixes=L2),
    # pick person randomly from all persons
    if len(possible_persons) == 0:
        possible_persons = persons

    # Choose person, weighted by how many affixes match received affix
    # (can be one possible person, so choice is trivial)
    inferred_person = RG.choice(possible_persons)
    return inferred_person


def infer_possible_persons(affix_type, affix_signal, persons, affixes, lex_concept):
    # Calculate distances of internal affixes to received affix,
    # to find candidate persons
    possible_persons = []
    for p in persons:
        affixes_person = affixes[(lex_concept, p, affix_type)]
        for affix_internal in affixes_person:
            # Add all internal affixes which exactly match signal
            if affix_internal == affix_signal:
                possible_persons.append(p)
    return possible_persons


def reduce_phonotactics(affix_type, affix, form, clts, cv_pattern_cache, drop_boundary_phoneme):
    if len(affix) > 0:
        stem_border = form[0] if affix_type == "prefix" else form[-1]
        affix_border = affix[-1] if affix_type == "prefix" else affix[0]
        spaced_border_seq = f"{affix_border} {stem_border}" if affix_type == "prefix" else f"{stem_border} {affix_border}"
        if spaced_border_seq in cv_pattern_cache:
            cv_pattern = cv_pattern_cache[spaced_border_seq]
        else:
            cv_pattern = clts.bipa.translate(spaced_border_seq, clts.soundclass("cv")).replace(" ", "")
            cv_pattern_cache[spaced_border_seq] = cv_pattern

        if cv_pattern == "CC":
            if drop_boundary_phoneme:
                if affix_type == "prefix":
                    affix = affix[:-1]
                elif affix_type == "suffix":
                    affix = affix[1:]
                else:
                    raise ValueError()
            else:
                # Drop whole affix
                affix = ""
    return affix


def update_affix_list(prefix_recv, suffix_recv, affixes, lex_concepts_type,
                      concept_listener):
    lex_concept_listener = concept_listener.lex_concept
    person_listener = concept_listener.person
    for affix_type, affix_recv in [("prefix", prefix_recv), ("suffix", suffix_recv)]:
        # NOTE: This assumes listener has same concept matrix, and knows which
        # verbs are prefixing/suffixing.
        if lex_concept_listener not in lex_concepts_type[affix_type]:
            # If no affix for this verb type received (e.g. prefix), skip
            continue
        # Normal update: do not generalize
        lex_concepts = [lex_concept_listener]
        persons = [person_listener]
        for lex_concept in lex_concepts:
            for person in persons:
                affix_list = affixes[(lex_concept, person, affix_type)]
                if affix_recv is None:
                    # NOTE: check should be unnecessary
                    raise Exception("Affix cannot be None, if this affix type is enabled for this verb!")
                # Positive update
                affix_list.append(affix_recv)


def spread_l2_agents(proportion_l2, n_agents):
    n_l2_agents = math.floor(proportion_l2 * n_agents)
    l2 = np.zeros(n_agents)
    l2[0:n_l2_agents] = 1
    RG.shuffle(l2)
    return [bool(x) for x in l2]


def weighted_affixes_prior_combined(lex_concept, person, affix_type, affixes):
    affixes_concept = affixes[(lex_concept, person, affix_type)]
    p_affix_given_concept, p_affix_given_concept_dict, _ = compute_affix_given_concept(affixes_concept)

    affixes_affix_type = {(l, p, t): affixes[(l, p, t)] for (l, p, t) in affixes.keys() if t == affix_type and (
        GENERALIZE_PERSONS or p == person) and (GENERALIZE_LEX_CONCEPTS or l == lex_concept)}
    affixes_all = list(chain.from_iterable(affixes_affix_type.values()))
    logging.debug(f"Affixes for all: {affixes_all}")
    n_exemplars_all = len(affixes_all)
    counts_affixes_all = Counter(affixes_all)
    logging.debug(f"Counts for all: {counts_affixes_all}")
    p_affix = {aff: count/n_exemplars_all for aff, count in counts_affixes_all.items()}
    logging.debug(f"Probabilities for all: {p_affix}")

    # combined: p(affix|concept) * p(affix) [prior]
    p_combined = {aff: p_aff_conc * p_affix[aff] for aff, p_aff_conc in p_affix_given_concept.items()}
    logging.debug(f"Combined: {p_combined}")
    total = sum(p_combined.values())
    p_combined_normalized = {aff: p_comb/total for aff, p_comb in p_combined.items()}
    logging.debug(f"Combined normalized: {p_combined_normalized}")
    return p_combined_normalized


def use_generalization(lex_concept, person, affix_type, affixes, generalization_prob=None):

    # With a certain probability, use distribution of all concepts
    # rest of times, use distribution of specific concept
    if RG.random() < generalization_prob:
        logging.debug(f"Use generalization.")
        if GENERALIZE_PREFIX_SUFFIX:
            affixes_affix_type = {(l, p, t): affixes[(l, p, t)] for (l, p, t) in affixes.keys() if (
                GENERALIZE_PERSONS or p == person) and (GENERALIZE_LEX_CONCEPTS or l == lex_concept)}
        else:
            affixes_affix_type = {(l, p, t): affixes[(l, p, t)] for (l, p, t) in affixes.keys() if t == affix_type and (
                GENERALIZE_PERSONS or p == person) and (GENERALIZE_LEX_CONCEPTS or l == lex_concept)}
        affixes_all = list(chain.from_iterable(affixes_affix_type.values()))
        _, p_affix_dict, _ = compute_affix_probabilities(affixes_all)
        return p_affix_dict
    else:
        # Same code as first lines of now disabled method distribution_from_exemplars
        affixes_concept = affixes[(lex_concept, person, affix_type)]
        _, p_affix_given_concept_dict, _ = compute_affix_probabilities(affixes_concept)
        logging.debug(f"Use concept distribution: {p_affix_given_concept_dict}")
        return p_affix_given_concept_dict


def compute_affix_probabilities(affixes):
    logging.debug(f"Affixes: {affixes}")

    n_exemplars = len(affixes)
    counts_affixes = Counter(affixes)
    logging.debug(f"Counts: {counts_affixes}")
    counts_keys = list(counts_affixes.keys())
    counts_values = np.array(list(counts_affixes.values()))
    p = counts_values / n_exemplars
    p_dict = dict(zip(counts_keys, p))
    logging.debug(f"Probabilities: {p_dict}")
    return p, p_dict, counts_keys


def affix_choice(affixes):
    if isinstance(affixes, dict):
        # Choice weighted by probability
        sample = RG.choice(list(affixes.keys()), p=list(affixes.values()))
    elif isinstance(affixes, list):
        # Unweighted choice
        sample = RG.choice(affixes)
    else:
        raise ValueError("affixes has to be a dict or list.")
    return sample


def create_output_dir(output_dir):
    # Check if dir exists only for test scripts,
    # in normal cases dir should be created once and not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

