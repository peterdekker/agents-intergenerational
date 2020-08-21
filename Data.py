import pandas as pd
import re

class Data():
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep="\t")
        self.concepts = list(self.data["concept"])
        self.persons = [col for col in self.data if re.match("[0-9]",col)]

        # Create transitivity dict
        

if __name__ == "__main__":
    d = Data("data/data.csv")
