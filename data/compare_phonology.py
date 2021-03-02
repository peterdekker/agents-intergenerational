
import common
import pandas as pd


def main():
    common.download_lexirumah()
    # Read lects
    lects_df = pd.read_csv(common.LECTS_PATH)
    lects_alorese = lects_df[lects_df["Name"].str.startswith("Alorese")]
    lects_lamaholot_related = lects_df[lects_df["Name"].str.contains(
        "Lamalera|Lewotobi|Lewoingu|Lewolema", case=False, regex=True)]
    lects_tap = lects_df[lects_df["Family"].str.startswith("Timor-Alor-Pantar")]

    # Read data
    data = pd.read_csv(common.DATA_PATH)
    # Join data with lect name from lects table
    data = data.merge(lects_df[["ID", "Name", "Family"]], how="left", left_on="Lect_ID", right_on="ID")
    print(data.head())
    for langs, label in [(lects_austro, "austronesian"), (lects_alorese, "alorese"), (lects_alorese_lamaholot, "alorese-lamaholot")]:
        data_langs = data[data["Lect_ID"].isin(langs["ID"])]


if __name__ == "__main__":
    main()
