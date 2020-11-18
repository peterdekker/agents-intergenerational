from constants import RG

# def random_choice_weighted_dict(weighted_dict):
#     # This functions randomly chooses from a dict
#     # where keys are the choices, and the values
#     # are the accompanying probabilities.
#     chosen_item = RG.choice(list(weighted_dict.keys()), p=list(weighted_dict.values()))
#     return chosen_item

def is_prefixing_verb(prefixes):
    return '' not in prefixes # TODO: only minimal check, what with more affixes?

def is_suffixing_verb(suffixes):
    return '' not in suffixes # TODO: only minimal check, what with more affixes?
