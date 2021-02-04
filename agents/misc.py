import editdistance
import math
import numpy as np

from agents.config import dst, logging, RG


def lookup_lex_concept(signal_form, lex_concepts, lex_concept_data):
    retrieved_lex_concept = None
    for lex_concept in lex_concepts:
        if lex_concept_data[lex_concept]["form"] == signal_form:
            retrieved_lex_concept = lex_concept
            break
    if not retrieved_lex_concept:
        raise Exception("Inferred lex concept cannot be None!")
    return retrieved_lex_concept


def infer_person_from_signal(lex_concept, lex_concept_data, affixes, persons, signal, ambiguity):
    logging.debug("Person not given via context, inferring from affix.")
    # TODO: This assumes listener has same concept matrix, and knows which
    # verbs are prefixing/suffixing

    possible_persons = []
    if lex_concept_data[lex_concept]["prefixing"]:
        possible_persons += infer_possible_persons("prefix", signal.get_prefix(), persons,
                                                   affixes, lex_concept, ambiguity)
    if lex_concept_data[lex_concept]["suffixing"]:
        possible_persons += infer_possible_persons("suffix",
                                                   signal.get_suffix(), persons,
                                                   affixes, lex_concept, ambiguity)

    # If no possible persons (because no affix, or empty internal suffixes=L2),
    # pick person randomly from all persons
    if len(possible_persons) == 0:
        possible_persons = persons

    # Choose person, weighted by how many affixes are closest to received affix
    # (can be one possible person, so choice is trivial)
    inferred_person = RG.choice(possible_persons)
    return inferred_person


def infer_possible_persons(affix_type, affix_signal, persons, affixes, lex_concept, ambiguity):
    # Calculate distances of internal affixes to received affix,
    # to find candidate persons
    possible_persons = []
    lowest_dist = 1000
    for p in persons:
        affixes_person = affixes[(lex_concept, p, affix_type)]
        for affix_internal in affixes_person:
            dist = editdistance.eval(affix_signal, affix_internal)
            if dist <= lowest_dist:
                if dist < lowest_dist:
                    lowest_dist = dist
                    possible_persons = []
                possible_persons.append(p)

    persons_ambig = len(possible_persons) if len(possible_persons) > 0 else len(persons)
    ambiguity[f"'{affix_signal}'-{affix_type}"].append(persons_ambig)
    return possible_persons


def reduce_affix_phonetic(verb_type, affix, form, min_boundary_feature_dist):
    form_border_phoneme = 0 if verb_type == "prefixing" else -1
    affix_border_phoneme = -1 if verb_type == "prefixing" else 0
    affix_slice = affix[affix_border_phoneme] if len(affix) > 0 else affix
    feature_dist = dst.weighted_feature_edit_distance(form[form_border_phoneme], affix_slice)
    # Sounds have to be different enough
    if feature_dist < min_boundary_feature_dist:
        affix = ""
    return affix


def reduce_affix_hh(verb_type, affix, listener, reduction_hh):
    affix_red = affix
    if reduction_hh:
        if not listener.is_l2():
            if len(affix) > 0:
                affix_red = affix[1:] if verb_type == "prefixing" else affix[:-1]
                logging.debug(f"H&H: {affix} -> {affix_red}")
    return affix_red


def enforce_capacity(affix_list, capacity):
    while len(affix_list) > capacity:
        affix_list.pop(0)


def update_affix_list(affix_type, affix_recv, affixes, lex_concept_data, lex_concept_listener, person_listener, capacity):
    affix_list = affixes[(lex_concept_listener, person_listener, affix_type)]
    if lex_concept_data[lex_concept_listener][f"{affix_type}ing"]:
        if affix_recv is None:
            # TODO: check should be unnecessary. delete later
            raise Exception("Affix cannot be None, if this affix type is enabled for this verb!")
        affix_list.append(affix_recv)
        logging.debug(
            f"{affix_type.capitalize()}es after update: {affix_list}")
        enforce_capacity(affix_list, capacity)


def spread_l2_agents(proportion_l2, n_agents):
    n_l2_agents = math.floor(proportion_l2 * n_agents)
    l2 = np.zeros(n_agents)
    l2[0:n_l2_agents] = 1
    RG.shuffle(l2)
    return l2
