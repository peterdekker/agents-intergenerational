import argparse
import pandas as pd

params_cols = ["proportion_l2", "capacity_l1", "capacity_l2", "generalize_update_l1", "generalize_update_l2"]


def analyze(args):
    lsuff = ""
    rsuff = ""
    df = pd.read_csv(args.file1, sep="\t", index_col=0)
    if args.file2:
        df, lsuff = join_dfs(args.file2, df, lsuff, rsuff)
    # Find most different values
    # df["l1_pre_suff_diff"] = df[f"prop_internal_prefix_l1{lsuff}"] - \
    #     df[f"prop_internal_suffix_l1{lsuff}"]
    # df = df.sort_values(by="l1_pre_suff_diff", ascending=False)
    df = df.sort_values(by=f"prop_internal_suffix_l1{lsuff}", ascending=False)
    df = df.loc[df['proportion_l2'].isin([0.0, 0.7])]
    print(df)
    df.to_csv("comb.tsv", sep="\t")


def join_dfs(file2, df, lsuff, rsuff):
    df2 = pd.read_csv(file2, sep="\t", index_col=0)
    lsuff = "_10k"
    rsuff = "_5k"
    # Remove duplicate columns (should this be needed?)
    for param_col in params_cols:
        del df2[param_col]
    df = df.join(other=df2, lsuffix=lsuff, rsuffix=rsuff)
    return df, lsuff


def main():
    parser = argparse.ArgumentParser(description='Results analysis script')
    parser.add_argument("--file1")
    parser.add_argument("--file2")
    # Parse arguments
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
