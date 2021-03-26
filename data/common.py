
from pycldf.dataset import Dataset
from itertools import chain
from nltk.util import ngrams
from collections import Counter
import os
import pathlib
import requests
import shutil
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [12, 6]


DATA_ARCHIVE_URL = "https://github.com/lessersunda/lexirumah-data/archive/v3.0.0.tar.gz"
DATA_ARCHIVE_PATH = "lexirumah.tar.gz"
DATA_UNPACK_PATH = "lexirumah-data-3.0.0"
METADATA_PATH = os.path.join(DATA_UNPACK_PATH, "cldf/cldf-metadata.json")
# DATA_PATH = os.path.join(DATA_UNPACK_PATH, "cldf/forms.csv")
# LECTS_PATH = os.path.join(DATA_UNPACK_PATH, "cldf/lects.csv")

USE_RAW_FREQ = False


def download_if_needed(file_path, url, label):
    if not os.path.exists(file_path):
        # Create parent dirs
        p = pathlib.Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            print(f"Downloading {label} from {url}")
            try:
                r = requests.get(url, allow_redirects=True)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            # Write downloaded content to file
            f.write(r.content)
            if file_path.endswith(".tar.gz"):
                print("Unpacking archive.")
                shutil.unpack_archive(file_path)


def load_lexirumah():
    download_if_needed(DATA_ARCHIVE_PATH, DATA_ARCHIVE_URL, "LexiRumah")
    print("Loading data...")
    dataset = Dataset.from_metadata(METADATA_PATH)
    data_df = pd.DataFrame(dataset["FormTable"])
    lects_df = pd.DataFrame(dataset["LanguageTable"])
    print("Loaded data.")
    return data_df, lects_df


def shared_feature_matrix(df, column_name):
    column = df[column_name]
    keys = column.apply(list)
    unique_keys = set(keys.aggregate(sum))
    lects = df.index
    shared_matrix = pd.DataFrame(index=lects, columns=unique_keys)
    for lect in lects:
        prob_dict = column.loc[lect]
        for key, prob in prob_dict.items():
            shared_matrix.at[lect, key] = prob
    return shared_matrix


def freq_to_prob(freq_dict):
    n_tokens = sum(freq_dict.values())  # faster than elements
    if USE_RAW_FREQ:
        return freq_dict
    return {k: v / n_tokens for k, v in freq_dict.items()}


def freq_to_existence(counter):
    return {k: int(v > 0) for k, v in counter.items()}


def phone_prob_existence(phone_lists):
    return phone_prob(phone_lists, existence=True)


def biphone_prob_existence(phone_lists):
    return biphone_prob(phone_lists, existence=True)


def phone_prob_boundaries(phone_lists):
    return phone_prob(phone_lists, word_boundaries=True)


def phone_prob(phone_lists, existence=False, word_boundaries=False):
    if word_boundaries:
        phone_lists = [['<']+x+['>'] for x in phone_lists]
    flattened = chain.from_iterable(phone_lists)
    counter = Counter(flattened)
    table = freq_to_existence(counter) if existence else freq_to_prob(counter)
    return table


# TODO: Make conditional probabilities, forward/backward
# Handle missing data (None for non-existent phonemes)
def biphone_prob(phone_lists, existence=False):
    counter = Counter()
    for phone_list in phone_lists:
        counter.update(ngrams(phone_list, 2, pad_left=True, pad_right=True,
                              left_pad_symbol='<', right_pad_symbol='>'))

    table = freq_to_existence(counter) if existence else freq_to_prob(counter)
    return table


def create_matrix(data_df, segments_col, calculation_function, label):
    data_df[label] = data_df[segments_col].apply(calculation_function)
    feature_matrix = shared_feature_matrix(data_df, label)
    return feature_matrix


def reduce_plot(study_label, study_data, dr_label, dr, data_agg, language_groups, plot_labels, reduce):
    study_data = study_data.fillna(0)
    if reduce:
        std_data = StandardScaler().fit_transform(study_data)
        red_data = dr.fit_transform(std_data)
        x = [i[0] for i in red_data]
        y = [i[1] for i in red_data]
        data_agg[f"{study_label}-{dr_label}-pc1"] = x
        data_agg[f"{study_label}-{dr_label}-pc2"] = y
    else:
        data_agg[f"{study_label}-{dr_label}-pc1"] = data_agg[dr[0]]
        data_agg[f"{study_label}-{dr_label}-pc2"] = data_agg[dr[1]]
    # Plot datapoints per language group, for correct label + color
    for lang_group in language_groups:
        lects_group = language_groups[lang_group]["lects"]
        color = language_groups[lang_group]["color"]
        data_group = data_agg[data_agg.index.isin(lects_group)]
        data_x = data_group[f"{study_label}-{dr_label}-pc1"]
        data_y = data_group[f"{study_label}-{dr_label}-pc2"]
        plt.scatter(data_x, data_y, color=color, label=lang_group, s=36)
        if plot_labels:
            for _, row in data_group.iterrows():
                plt.annotate(row.name, xy=(row[f"{study_label}-{dr_label}-pc1"],
                                           row[f"{study_label}-{dr_label}-pc2"]))
    plt.title(f"{study_label} ({dr_label})")
    plt.legend()
    plt.show()


def compute_loadings(dr, feature_names):
    loadings = pd.DataFrame(dr.components_.T, columns=['PC1', 'PC2'], index=feature_names)
    #loadings = loadings.abs()
    NL = 10
    loadings_x_pos = loadings.sort_values(by="PC1", ascending=False)["PC1"].head(NL)
    loadings_x_neg = loadings.sort_values(by="PC1", ascending=True)["PC1"].head(NL)
    loadings_y_pos = loadings.sort_values(by="PC2", ascending=False)["PC2"].head(NL)
    loadings_y_neg = loadings.sort_values(by="PC2", ascending=True)["PC2"].head(NL)
    return loadings_x_pos, loadings_x_neg, loadings_y_pos, loadings_y_neg
