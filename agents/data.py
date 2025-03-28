import pandas as pd
import re

from agents.config import PERSONS


class Data():
    def __init__(self, data_file, interaction_l1, interaction_l1_shield_initialization):
        # unique_affix_id = 0
        self.data = pd.read_csv(data_file, sep="\t").fillna(value="")
        # Filter on only cells which have Lewoingu form
        self.data = self.data[self.data["form_lewoingu"] != ""]

        # if balance_prefix_suffix_verbs:
        #     prefixing = self.data[self.data["prefix"] == 1]
        #     n_prefixing = len(prefixing)
        #     suffixing = self.data[self.data["suffix"] == 1]
        #     self.data = pd.concat([prefixing, suffixing.head(n_prefixing)])

        self.lex_concepts = list(self.data["concept"])
        self.lex_concepts_type = {"prefix": [], "suffix": []}
        self.persons = PERSONS

        general_cols = ["concept", "form_lewoingu", "prefix", "suffix"]
        person_affix_cols = [col for col in self.data if col.startswith(tuple(PERSONS))]
        data_reindexed = self.data[general_cols+person_affix_cols].set_index("concept")
        data_dict = data_reindexed.to_dict(orient="index")
        self.lex_concept_data = {}
        self.affixes = {}

        for lex_concept in self.lex_concepts:
            # TODO: possible to do all these transformations vectorized
            # in df and convert to dict?
            # Form processing: split multiple forms on , or /
            # strip * and -
            # Just use first form
            forms = data_dict[lex_concept]["form_lewoingu"]
            form = [f.strip("* -") for f in re.split(",|/", forms)][0]
            prefixing = bool(data_dict[lex_concept]["prefix"])
            suffixing = bool(data_dict[lex_concept]["suffix"])
            # For now, make it not possible for prefixing verbs to be also suffixing
            if prefixing:
                suffixing = False
                self.lex_concepts_type["prefix"].append(lex_concept)
            else:
                self.lex_concepts_type["suffix"].append(lex_concept)
            self.lex_concept_data[lex_concept] = {"form": form,
                                                  "prefix": prefixing,
                                                  "suffix": suffixing}

            for person in self.persons:
                for affix_type in ["prefix", "suffix"]:
                    # For now, make it not possible for prefixing verbs to be also suffixing
                    # NOTE: check not very sophisticated. we just filled this lex_concepts_type
                    if lex_concept in self.lex_concepts_type[affix_type]:
                        affix = data_dict[lex_concept][f"{person}_{affix_type}"]
                        # Affix preprocessing: there can be multiple affixes, split by ,
                        affixes_processed = [a.strip(" -") for a in affix.split(",")]
                        if "" in affixes_processed:
                            affixes_processed.remove("")

                        # Double affix if there is only 1 affix. So length of list is always 2,
                        # regardless if there are 1 or 2 variants.
                        if len(affixes_processed) == 1:
                            affixes_processed *= 2
                        # Only if interaction_l1 is on, L1 initialization is shielded
                        repeat_list = interaction_l1_shield_initialization if interaction_l1 else 1
                        self.affixes[(lex_concept, person, affix_type)] = affixes_processed * repeat_list

