import pandas as pd
import re
from collections import defaultdict
from constants import PERSONS

class Data():
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep="\t").fillna(value="")
        self.concepts = list(self.data["concept"])
        self.persons = PERSONS

        # Create transitivity dict: concept -> {trans:0/1, instrans: 0/1}
        data_trans = self.data[["concept","trans","intrans"]].set_index("concept")
        self.transitivities = data_trans.to_dict(orient="index")

        # Create forms dict: concept -> {form_alorese: x, form_lewoingu, y}
        data_forms = self.data[["concept","form_alorese", "form_lewoingu"]].set_index("concept")
        data_forms_dict = data_forms.to_dict(orient="index")
        self.forms = defaultdict(dict)

        # Create affixes dict: concept -> (person -> affix)
        # This will be used as basis for the mental models of the agents
        person_affix_cols = [col for col in self.data if re.match("[0-9]",col)]
        data_affixes = self.data[["concept"]+person_affix_cols].set_index("concept")
        data_affixes_dict = data_affixes.to_dict(orient="index")
        self.affixes = defaultdict(lambda: defaultdict(dict))

        for concept in self.concepts:
            # Form processing: split multiple forms on , or /
            # strip * and -
            # TODO: Find out where variation in verb forms comes from
            for lang in ["alorese", "lewoingu"]:
                form = data_forms_dict[concept][f"form_{lang}"]
                # If no form for this language, dont add key
                if form == "":
                    continue
                forms_processed = [f.strip("*-") for f in re.split(",|/", form)]
                # Build dict of possible affixes for this type, with uniform distribution
                form_prob = 1.0/len(forms_processed)
                form_prob_dict = {f: form_prob for f in forms_processed}
                self.forms[concept][lang] = form_prob_dict
            for person in self.persons:
                for affix_type in ["prefix", "suffix"]:
                    affix = data_affixes_dict[concept][f"{person}_{affix_type}"]
                    # Affix preprocessing: there can be multiple affixes, split by ,
                    affixes_processed = [a.strip(" -") for a in affix.split(",")]
                    # Build dict of possible affixes for this type, with uniform distribution
                    aff_prob = 1.0/len(affixes_processed)
                    aff_prob_dict = {aff: aff_prob for aff in affixes_processed}

                    self.affixes[concept][person][affix_type] = aff_prob_dict
        print(self.affixes)
        print(self.forms)
            

        # check double items: print([item for item, count in collections.Counter(self.concepts).items() if count > 1])


if __name__ == "__main__":
    d = Data("data/data.csv")

