
import common
import pandas as pd


def main():
    common.download_lexirumah()
    # Read lects
    lects_df = pd.read_csv(common.LECTS_PATH)
    lects_alorese = lects_df[lects_df["Name"].str.startswith("Alorese")]
    lects_austro = lects_df[lects_df["Family"].str.startswith("Austronesian")]
    lects_lamaholot_related = lects_df[lects_df["Name"].str.contains(
        "Lamalera|Lewotobi|Lewoingu|Lewolema", case=False, regex=True)]
    lects_alorese_lamaholot = lects_alorese.append(lects_lamaholot_related)
    print(f"Total number of lects: {len(lects_df)}")
    print(f"Number of lects Austronese: {len(lects_austro)}")
    print(f"Number of lects Alorese: {len(lects_alorese)}")
    print(f"Number of lects Lamaholot lects related to Alorese: {len(lects_lamaholot_related)}")

    # Read data
    data = pd.read_csv(common.DATA_PATH)
    # Math only personal pronouns, starting with number (eg. 1sg)
    data = data[data["Concept_ID"].str.match(r"^\d.*")]
    # Join data with lect name from lects table
    data = data.merge(lects_df[["ID", "Name"]], how="left", left_on="Lect_ID", right_on="ID")
    for langs, label in [(lects_austro, "austronesian"), (lects_alorese, "alorese"), (lects_alorese_lamaholot, "alorese-lamaholot")]:
        data_langs = data[data["Lect_ID"].isin(langs["ID"])]
        data_new = data_langs.pivot_table(index='Concept_ID', columns='Name', values='Form', aggfunc="/".join)
        order = ['1sg', '2sg_informal', '2sg_polite', '3sg', '1pl_excl', '1pl_incl', '2pl', '3pl']
        data_new = data_new.loc[order]
        data_new.to_csv(f"paradigm_{label}.tsv", sep="\t")


if __name__ == "__main__":
    main()
