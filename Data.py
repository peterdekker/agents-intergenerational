import pandas as pd
import re
from collections import defaultdict
from constants import PERSONS


class Data():
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep="\t").fillna(value="")
        # Filter on only entries which have Lewoingu form (1 is not. TODO: fix data)
        self.data = self.data[self.data["form_lewoingu"] != ""]
        self.lex_concepts = list(self.data["concept"])
        self.persons = PERSONS

        # Create transitivity dict: lex_concept -> {trans:0/1, instrans: 0/1}
        data_trans = self.data[["concept", "trans", "intrans"]].set_index("concept")
        self.transitivities = data_trans.to_dict(orient="index")
        for lex_concept in self.lex_concepts:
            self.transitivities[lex_concept] = [
                k for k, v in self.transitivities[lex_concept].items() if v == 1]

        # Create forms dict: lex_concept -> form
        data_forms = self.data[["concept", "form_lewoingu"]].set_index("concept")
        data_forms_dict = data_forms.to_dict(orient="index")
        self.forms = defaultdict(str)

        # Create affixes dict: (concept,person,affix_type)->list of affixes
        # This will be used as basis for the mental models of the agents
        person_affix_cols = [col for col in self.data if re.match("[0-9]", col)]
        data_affixes = self.data[["concept"]+person_affix_cols].set_index("concept")
        data_affixes_dict = data_affixes.to_dict(orient="index")
        self.affixes = defaultdict(list)

        for lex_concept in self.lex_concepts:
            # Form processing: split multiple forms on , or /
            # strip * and -
            form = data_forms_dict[lex_concept]["form_lewoingu"]
            # If no form for this language, dont add key
            if form == "":
                continue
            forms_processed = [f.strip("* -") for f in re.split(",|/", form)]
            # Just use the first form
            self.forms[lex_concept] = forms_processed[0]
            for person in self.persons:
                for affix_type in ["prefix", "suffix"]:
                    affix = data_affixes_dict[lex_concept][f"{person}_{affix_type}"]
                    # Affix preprocessing: there can be multiple affixes, split by ,
                    affixes_processed = [a.strip(" -") for a in affix.split(",")]
                    if "" in affixes_processed:
                        affixes_processed.remove("")
                    #aff_prob = 1.0/len(affixes_processed)
                    #aff_prob_dict = {aff: aff_prob for aff in affixes_processed}
                    self.affixes[(lex_concept, person, affix_type)] = affixes_processed

        # check double items: print([item for item, count in collections.Counter(self.concepts).items() if count > 1])


# if __name__ == "__main__":
#     d = Data(DATA_FILE)
