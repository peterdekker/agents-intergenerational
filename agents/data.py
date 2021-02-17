import pandas as pd
import re

from agents.config import PERSONS


class Data():
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep="\t").fillna(value="")
        # Filter on only cells which have Lewoingu form
        self.data = self.data[self.data["form_lewoingu"] != ""]
        self.lex_concepts = list(self.data["concept"])
        self.lex_concepts_type = {"prefixing": [], "suffixing": []}
        self.persons = PERSONS

        general_cols = ["concept", "form_lewoingu", "trans", "intrans", "prefixing", "suffixing"]
        person_affix_cols = [col for col in self.data if col.startswith(tuple(PERSONS))]
        data_reindexed = self.data[general_cols+person_affix_cols].set_index("concept")
        data_dict = data_reindexed.to_dict(orient="index")
        self.lex_concept_data = {}
        self.affixes = {}  # defaultdict(list)

        for lex_concept in self.lex_concepts:
            # TODO: possible to do all these transformations vectorized
            # in df and convert to dict?
            # Form processing: split multiple forms on , or /
            # strip * and -
            # Just use first form
            forms = data_dict[lex_concept]["form_lewoingu"]
            form = [f.strip("* -") for f in re.split(",|/", forms)][0]
            transitivities = [k for k, v in data_dict[lex_concept].items() if (
                k == "trans" or k == "intrans") and v == 1]
            prefixing = bool(data_dict[lex_concept]["prefixing"])
            suffixing = bool(data_dict[lex_concept]["suffixing"])
            # For now, make it not possible for prefixing verbs to be also suffixing
            if prefixing:
                suffixing = False
                self.lex_concepts_type["prefixing"].append(lex_concept)
            else:
                self.lex_concepts_type["suffixing"].append(lex_concept)
            self.lex_concept_data[lex_concept] = {"form": form,
                                                  "transitivities": transitivities,
                                                  "prefixing": prefixing,
                                                  "suffixing": suffixing}

            for person in self.persons:
                for affix_type in ["prefix", "suffix"]:
                    affix = data_dict[lex_concept][f"{person}_{affix_type}"]
                    # Affix preprocessing: there can be multiple affixes, split by ,
                    affixes_processed = [a.strip(" -") for a in affix.split(",")]
                    if "" in affixes_processed:
                        affixes_processed.remove("")
                    self.affixes[(lex_concept, person, affix_type)] = affixes_processed

        # check double items: print([item for item, count in collections.Counter(self.concepts).items() if count > 1])


# if __name__ == "__main__":
#     d = Data(DATA_FILE)
