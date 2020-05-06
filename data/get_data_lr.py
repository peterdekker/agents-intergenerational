import pandas as pd
import time
DATA_PATH = "../../lexirumah-data/cldf/forms.csv"
LECTS_PATH = "../../lexirumah-data/cldf/lects.csv"
start = time.time()
### Read lects
lects_df = pd.read_csv(LECTS_PATH)
lects_alorese = lects_df[lects_df["Name"].str.startswith("Alorese")]
lects_austro = lects_df[lects_df["Family"].str.startswith("Austronesian")]
print(f"Total number of lects: {len(lects_df)}")
print(f"Number of lects Alorese: {len(lects_alorese)}")
print(f"Number of lects Austronese: {len(lects_austro)}")

### Read data
data = pd.read_csv(DATA_PATH)
# Math only personal pronouns, starting with number (eg. 1sg)
data = data[data["Concept_ID"].str.match(r"^\d.*")]
# Join data with lect name from lects table
data = data.merge(lects_df[["ID","Name"]], how="left", left_on="Lect_ID", right_on="ID")
print(data)
for langs, label in [(lects_alorese, "alorese"), (lects_austro, "austronesian")]:
    data_langs = data[data["Lect_ID"].isin(langs["ID"])]
    data_new = data_langs.pivot_table(index='Concept_ID', columns='Name', values='Form', aggfunc="/".join)
    order = ['1sg', '2sg_informal', '2sg_polite', '3sg', '1pl_excl', '1pl_incl', '2pl', '3pl' ]
    #data_new = data_new.loc[order]
    data_new.to_csv(f"paradigm_{label}.tsv", sep="\t")

end=time.time()
print(end-start)


#data_new = pd.pivot_table(data=data_austro[["Concept_ID", "Lect_ID", "Form"]],index="Concept_ID", columns="Lect_ID", values="Form")
#print(data_new)
# grouped_df = data_alorese[["Lect_ID","Concept_ID","Form"]].groupby("Lect_ID")
# first = True
# for key, item in grouped_df:
#     gdf = grouped_df.get_group(key)
#     gdf_reindex = gdf.set_index("Concept_ID")
#     gdf_changed = gdf_reindex.rename(columns={"Form":key}).drop(columns=["Lect_ID"])
#     if first:
#         df_combined = gdf_changed
#         first = False
#     else:
#         df_combined = df_combined.join(gdf_changed, how="outer")
# print(df_combined)