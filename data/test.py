import pandas as pd

df = pd.read_csv("test.csv")
print(df)
df = df.set_index(['lect', 'concept', df.groupby(['lect', 'concept']).cumcount()])
print(df)
df = df.unstack('lect')
print(df)