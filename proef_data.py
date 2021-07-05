from agents.config import DATA_FILE
from agents.data import Data

if __name__ == "__main__":
    d = Data(DATA_FILE, balance_prefix_suffix_verbs=True)
