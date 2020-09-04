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

        # Create forms dict: concept -> {form_alorese: x, form_lmh, y}
        data_forms = self.data[["concept","form_alorese", "form_lmh"]].set_index("concept")
        self.forms = data_forms.to_dict(orient="index")

        # Create affixes dict: concept -> (person -> affix)
        # This will be used as basis for the mental models of the agents
        person_affix_cols = [col for col in self.data if re.match("[0-9]",col)]
        data_affixes = self.data[["concept"]+person_affix_cols].set_index("concept")
        data_affixes_dict = data_affixes.to_dict(orient="index")
        self.affixes = defaultdict(lambda : defaultdict(dict))
        for concept in self.concepts:
            for person in self.persons:
                for affix_type in ["prefix", "suffix"]:
                    self.affixes[concept][person][affix_type] = data_affixes_dict[concept][f"{person}_{affix_type}"]
        print(self.affixes)
        print(self.affixes["to eat"]["1pl.excl"]["suffix"])
            

        # check double items: print([item for item, count in collections.Counter(self.concepts).items() if count > 1])


if __name__ == "__main__":
    d = Data("data/data.csv")

