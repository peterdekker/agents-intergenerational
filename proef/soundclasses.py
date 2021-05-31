import pandas as pd
soundclass_systems = ["sca", "art", "dolgo", "asjp"]

df = pd.read_csv("lingpy.tsv", sep="\t", index_col="BIPA_GRAPHEME")
for sc in soundclass_systems:
    print(sc)
    col = df[sc]
    length = len(col.unique())
    freqs = col.value_counts()
    print(f"Number of symbols: {length}")
    print(freqs)
    print()
