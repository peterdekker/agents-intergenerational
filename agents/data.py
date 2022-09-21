import pandas as pd
import re

from agents.config import PERSONS


class Data():
    def __init__(self, data_file):
        unique_affix_id = 0
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
                    # TODO: make this check more sophisticated. we just filled this lex_concepts_type
                    if lex_concept in self.lex_concepts_type[affix_type]:
                        affix = data_dict[lex_concept][f"{person}_{affix_type}"]
                        # Affix preprocessing: there can be multiple affixes, split by ,
                        affixes_processed = [a.strip(" -") for a in affix.split(",")]
                        if "" in affixes_processed:
                            affixes_processed.remove("")
                        ###
                        # if len(affixes_processed) > 1:
                        #     affixes_processed = [affixes_processed[0]]
                        ###
                        # if unique_affix:
                        #     affixes_processed_unique = []
                        #     for a in affixes_processed:
                        #         affixes_processed_unique.append(a+str(unique_affix_id))
                        #         unique_affix_id += 1
                        #     affixes_processed = affixes_processed_unique

                        self.affixes[(lex_concept, person, affix_type)] = affixes_processed

        #print(f"Number of concept cells: {len(self.affixes.keys())}")
        # check double items: print([item for item, count in collections.Counter(self.concepts).items() if count > 1])
