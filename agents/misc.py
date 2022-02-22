import editdistance
import math
import numpy as np
import os
from itertools import chain

from agents.config import dst, logging, RG, CURRENTDIR, GENERALIZE_LEX_CONCEPTS, GENERALIZE_PERSONS

from collections import Counter

# Returns list


from pyclts import CLTS
import requests
import shutil


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
    if lex_concept_data[lex_concept]["prefixing"]:
        possible_persons += infer_possible_persons("prefix", signal.prefix, persons,
                                                   affixes, lex_concept)
    if lex_concept_data[lex_concept]["suffixing"]:
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


# def reduce_boundary_feature_dist(verb_type, affix, form, min_boundary_feature_dist, listener):
#     if min_boundary_feature_dist > 0.0:
#         form_border_phoneme = 0 if verb_type == "prefixing" else -1
#         affix_border_phoneme = -1 if verb_type == "prefixing" else 0
#         affix_slice = affix[affix_border_phoneme] if len(affix) > 0 else affix
#         feature_dist = dst.weighted_feature_edit_distance(form[form_border_phoneme], affix_slice)
#         # Sounds have to be different enough
#         if feature_dist < min_boundary_feature_dist:
#             affix = ""
#             # if len(affix) > 0:
#             #     affix = affix[1:] if verb_type == "prefixing" else affix[:-1]
#     return affix

# def reduce_hh(verb_type, affix, listener, reduction_hh):
#     #affix_old = affix
#     if reduction_hh:
#         if not listener.is_l2():
#             if len(affix) > 0:
#                 # affix = ""
#                 affix = affix[1:] if verb_type == "prefixing" else affix[:-1]
#                 # logging.debug(f"H&H: {affix_old} -> {affix}")
#     return affix

def reduce_prosody(verb_type, affix, form, reduction_prosody, listener, clts):
    if reduction_prosody:
        # form_border_phoneme = 0 if verb_type == "prefixing" else -1
        # affix_border_phoneme = -1 if verb_type == "prefixing" else 0
        # affix_slice = affix[affix_border_phoneme] if len(affix) > 0 else affix
        # feature_dist = dst.weighted_feature_edit_distance(form[form_border_phoneme], affix_slice)
        # # Sounds have to be different enough
        # if feature_dist < min_boundary_feature_dist:
        #     affix = ""
        #     # if len(affix) > 0:
        #     #     affix = affix[1:] if verb_type == "prefixing" else affix[:-1]
        
        inflected_form = affix+form if verb_type == "prefixing" else form+affix
        #print(inflected_form)
        spaced_form = " ".join(list(inflected_form))
        cv_pattern = clts.bipa.translate(spaced_form, clts.soundclass("cv")).replace(" ", "")
        #print(cv_pattern)
        if "CC" in cv_pattern:
            #print("CONSONANT CLUSTER!")
            affix = ""
    return affix


def enforce_capacity(affix_list, capacity):
    while len(affix_list) > capacity:
        affix_list.pop(0)


def update_affix_list(prefix_recv, suffix_recv, affixes, lex_concepts_type, lex_concept_data, persons_all,
                      concept_listener, capacity, generalize_update, l2, negative=False):
    lex_concept_listener = concept_listener.lex_concept
    person_listener = concept_listener.person
    for affix_type, affix_recv in [("prefix", prefix_recv), ("suffix", suffix_recv)]:
        # TODO: This assumes listener has same concept matrix, and knows which
        # verbs are prefixing/suffixing.
        if lex_concept_listener not in lex_concepts_type[f"{affix_type}ing"]: #affix_recv is None:
            # If no affix for this verb type received (e.g. prefix), skip
            continue
        if RG.random() < generalize_update:
            # Generalization: update all concepts
            lex_concepts = lex_concepts_type[f"{affix_type}ing"] if GENERALIZE_LEX_CONCEPTS else [lex_concept_listener]
            persons = persons_all if GENERALIZE_PERSONS else [person_listener]
        else:
            # Normal update: do not generalize
            lex_concepts = [lex_concept_listener]
            persons = [person_listener]
        for lex_concept in lex_concepts:
            for person in persons:
                affix_list = affixes[(lex_concept, person, affix_type)]
                if affix_recv is None:
                    # TODO: check should be unnecessary. delete later
                    raise Exception("Affix cannot be None, if this affix type is enabled for this verb!")
                if negative:
                    # Negative update
                    if affix_recv in affix_list:
                        # affix_list = [x for x in affix_list if x != affix_recv]
                        affix_list.remove(affix_recv)
                else:
                    # Positive update
                    affix_list.append(affix_recv)
                if capacity != 0: # capacity 0 means: do not enforce capacity
                    enforce_capacity(affix_list, capacity)


def spread_l2_agents(proportion_l2, n_agents):
    n_l2_agents = math.floor(proportion_l2 * n_agents)
    l2 = np.zeros(n_agents)
    l2[0:n_l2_agents] = 1
    RG.shuffle(l2)
    return l2


def retrieve_affixes_generalize(lex_concept, person, verb_type, affixes, generalize_production, lex_concepts,
                                persons, lex_concept_data):
    # Generalize: draw other concept to use affixes from
    if RG.random() < generalize_production:
        # Generalize: create list of all affixes, regardless of concept (but taking into account verb type)

        # Commented out code: draw one specific other concept
        # _, lex_concept_gen, person_gen, _ = ConceptMessage.draw_new_concept(lex_concepts,
        #                                                                     persons,
        #                                                                     lex_concept_data)
        # return affixes[(lex_concept_gen, person_gen, verb_type)]

        affixes_verb_type = {(l, p, t): affixes[(l, p, t)] for (l, p, t) in affixes.keys() if t == verb_type and (GENERALIZE_PERSONS or p == person) and (GENERALIZE_LEX_CONCEPTS or l == lex_concept)}
        affixes_all = list(chain.from_iterable(affixes_verb_type.values()))

        return affixes_all
    else:
        # Do not generalize: take affixes list for this concept
        return affixes[(lex_concept, person, verb_type)]


def affix_choice(affixes):
    return RG.choice(affixes)


def create_output_dir(output_dir):
    # Check if dir exists only for test scripts,
    # in normal cases dir should be created once and not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
