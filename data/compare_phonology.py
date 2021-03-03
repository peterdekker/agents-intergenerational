
import common


def main():
    data_df, lects_df = common.load_lexirumah()
    print(data_df.head())
    print(lects_df.head())
    # Choose lects
    lects_alorese = lects_df[lects_df["Name"].str.startswith("Alorese")]
    lects_lamaholot_related = lects_df[lects_df["Name"].str.contains(
        "Lamalera|Lewotobi|Lewoingu|Lewolema", case=False, regex=True)]
    lects_tap = lects_df[lects_df["Family"].str.startswith("Timor-Alor-Pantar")]
    lects = lects_alorese + lects_lamaholot_related + lects_tap

    # Join data with lect name from lects table
    data_df = data_df.merge(lects_df[["ID", "Name", "Family"]], how="left", left_on="Lect_ID", right_on="ID")
    print(data_df.head())
    data_langs = data_df[data_df["Lect_ID"].isin(lects["ID"])]


if __name__ == "__main__":
    main()
