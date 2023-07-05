# import editdistance
import math
import numpy as np
import os
from itertools import chain

from agents.config import logging, RG, CURRENTDIR, GENERALIZE_PERSONS, GENERALIZE_LEX_CONCEPTS

from collections import Counter

# Returns list


from pyclts import CLTS
import requests
import shutil
import math


def download_if_needed(archive_path, archive_url, file_path, label):
    if not os.path.exists(file_path):
        # Create parent dirs
        #p = pathlib.Path(file_path)
        #p.parent.mkdir(parents=True, exist_ok=True)
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


def load_clts(clts_archive_path, clts_archive_url, clts_path):
    # Download CLTS
    download_if_needed(clts_archive_path, clts_archive_url, clts_path, "CLTS")
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
    # TODO: This assumes listener has same concept matrix, and knows which
    # verbs are prefixing/suffixing

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

    # Choose person, weighted by how many affixes are closest to received affix
    # (can be one possible person, so choice is trivial)
    # possible_persons = list(set(possible_persons))
    inferred_person = RG.choice(possible_persons)
    return inferred_person


# , ambiguity):
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
    #persons_ambig = len(possible_persons) if len(possible_persons) > 0 else len(persons)
    # ambiguity[f"'{affix_signal}'-{affix_type}"].append(persons_ambig)
    return possible_persons


def reduce_phonotactics(affix_type, affix, form, clts, drop_border_phoneme):
    inflected_form = affix+form if affix_type == "prefix" else form+affix
    spaced_form = " ".join(list(inflected_form))
    cv_pattern = clts.bipa.translate(spaced_form, clts.soundclass("cv")).replace(" ", "")
    # print(f"{inflected_form} | {cv_pattern}")
    # print(cv_pattern)
    if "CC" in cv_pattern:
        # print("CONSONANT CLUSTER!")
        if drop_border_phoneme:
            # print(f"before: {affix}")
            if affix_type == "prefix":
                affix = affix[:-1]
            elif affix_type == "suffix":
                affix = affix[1:]
            else:
                raise ValueError()
            # print(f"after {affix}")
        else:
            # Drop whole affix
            affix = ""
    return affix


def update_affix_list(prefix_recv, suffix_recv, affixes, lex_concepts_type,
                      concept_listener):
    lex_concept_listener = concept_listener.lex_concept
    person_listener = concept_listener.person
    for affix_type, affix_recv in [("prefix", prefix_recv), ("suffix", suffix_recv)]:
        # TODO: This assumes listener has same concept matrix, and knows which
        # verbs are prefixing/suffixing.
        if lex_concept_listener not in lex_concepts_type[affix_type]:  # affix_recv is None:
            # If no affix for this verb type received (e.g. prefix), skip
            continue
        # Normal update: do not generalize
        lex_concepts = [lex_concept_listener]
        persons = [person_listener]
        for lex_concept in lex_concepts:
            for person in persons:
                affix_list = affixes[(lex_concept, person, affix_type)]
                if affix_recv is None:
                    # TODO: check should be unnecessary. delete later
                    raise Exception("Affix cannot be None, if this affix type is enabled for this verb!")
                # Positive update
                affix_list.append(affix_recv)


def spread_l2_agents(proportion_l2, n_agents):
    n_l2_agents = math.floor(proportion_l2 * n_agents)
    l2 = np.zeros(n_agents)
    l2[0:n_l2_agents] = 1
    RG.shuffle(l2)
    return [bool(x) for x in l2]


def weighted_affixes_prior(lex_concept, person, affix_type, affixes, mode, affix_prior_prob=None):
    affixes_concept = affixes[(lex_concept, person, affix_type)]
    logging.debug(f"Affixes for concept: {affixes_concept}")
    n_exemplars_concept = len(affixes_concept)
    counts_affixes_concept = Counter(affixes_concept)
    logging.debug(f"Counts for concept: {counts_affixes_concept}")
    # Division maybe more efficient as numpy array (see distribution_from_exemplars), but we need dict later
    p_affix_given_concept = {aff: count/n_exemplars_concept for aff, count in counts_affixes_concept.items()}
    logging.debug(f"Probabilities for concept: {p_affix_given_concept}")

    affixes_affix_type = {(l, p, t): affixes[(l, p, t)] for (l, p, t) in affixes.keys() if t == affix_type and (
        GENERALIZE_PERSONS or p == person) and (GENERALIZE_LEX_CONCEPTS or l == lex_concept)}
    affixes_all = list(chain.from_iterable(affixes_affix_type.values()))
    logging.debug(f"Affixes for all: {affixes_all}")
    n_exemplars_all = len(affixes_all)
    counts_affixes_all = Counter(affixes_all)
    logging.debug(f"Counts for all: {counts_affixes_all}")
    p_affix = {aff: count/n_exemplars_all for aff, count in counts_affixes_all.items()}
    logging.debug(f"Probabilities for all: {p_affix}")

    if mode == "only":
        # With a certain probability, use distribution of all concepts
        # rest of times, use distribution of specific concept
        logging.debug("Only affix prior mode.")
        if RG.random() < affix_prior_prob:
            logging.debug(f"Use affix prior: {p_affix}")
            return p_affix
        else:
            logging.debug(f"Use concept distribution: {p_affix_given_concept}")
            return p_affix_given_concept
    # Combine affix prior for all concepts and distribution of specific concept
    elif mode == "combined":
        logging.debug("Combined mode.")
        # combined: p(affix|concept) * p(affix) [prior]
        p_combined = {aff: p_aff_conc * p_affix[aff] for aff, p_aff_conc in p_affix_given_concept.items()}
        logging.debug(f"Combined: {p_combined}")
        total = sum(p_combined.values())
        p_combined_normalized = {aff: p_comb/total for aff, p_comb in p_combined.items()}
        logging.debug(f"Combined normalized: {p_combined_normalized}")
        return p_combined_normalized
    else:
        raise ValueError("Mode not recognized.")


def distribution_from_exemplars(lex_concept, person, affix_type, affixes, alpha):
    affixes_concept = affixes[(lex_concept, person, affix_type)]
    logging.debug(f"Affixes for concept: {affixes_concept}")
    if len(affixes_concept) == 0:
        return {}

    n_exemplars_concept = len(affixes_concept)
    counts_affixes_concept = Counter(affixes_concept)
    logging.debug(f"Counts for concept: {counts_affixes_concept}")
    counts_keys = list(counts_affixes_concept.keys())
    counts_values = np.array(list(counts_affixes_concept.values()))
    p_affix_given_concept = counts_values / n_exemplars_concept
    p_affix_given_concept_dict = dict(zip(counts_keys, p_affix_given_concept))
    logging.debug(f"Probabilities for concept: {p_affix_given_concept_dict}")
    logging.debug(f"Alpha: {alpha}")

    # If alpha is 1.0: stop calculation here and return probabilities.
    if math.isclose(alpha, 1.0):
        return p_affix_given_concept_dict

    # Scale probabilities with alpha in log space (make distribution peakier)
    log_scaled = np.log(p_affix_given_concept) * alpha
    logging.debug(f"Log Scaled: {dict(zip(counts_keys, log_scaled))}")
    log_moved = log_scaled - np.max(log_scaled)  # Move highest value to 0
    logging.debug(f"Log Moved: {dict(zip(counts_keys, log_moved))}")
    # Go back to normal probabilities
    p_moved = np.exp(log_moved)
    logging.debug(f"Prob Moved: {dict(zip(counts_keys, p_moved))}")
    p_scaled_normalized = p_moved / np.sum(p_moved)
    # if math.isclose(total, 0) and len(p_scaled) > 0:
    #     print(counts_affixes_concept)
    #     print(p_affix_given_concept)
    #     print(p_scaled)
    logging.debug(f"Scaled normalized: {dict(zip(counts_keys, p_scaled_normalized))}")
    return dict(zip(counts_keys, p_scaled_normalized))


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
