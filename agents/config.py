import numpy as np
import logging
import sys
import datetime
import os
from panphon.distance import Distance
from mesa.visualization.UserParam import UserSettableParameter, Slider, Checkbox

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

CURRENTDIR = os.path.dirname(os.path.realpath(__file__))
CLTS_ARCHIVE_PATH = os.path.join(CURRENTDIR, "2.1.0.tar.gz")
CLTS_ARCHIVE_URL = "https://github.com/cldf-clts/clts/archive/refs/tags/v2.1.0.tar.gz"
CLTS_PATH = os.path.join(CURRENTDIR, "clts-2.1.0")

HEIGHT = 8
WIDTH = 8
MAX_RADIUS = max(HEIGHT, WIDTH)

N_AGENTS = HEIGHT*WIDTH
SAMPLE = HEIGHT
DATA_FILE = "data/data-sample.csv"
DATA_FILE_SYNTHETIC = "data/data-syntheticforms-sample.csv"
PERSONS = ['1sg', '2sg', '3sg', '1pl.incl', '1pl.excl', '2pl', '3pl']

dst = Distance()
RG = np.random.default_rng()

COMMUNICATED_STATS_AFTER_STEPS = 1
RARE_STATS_AFTER_STEPS = 1000
COMM_SUCCESS_AFTER_STEPS = 50
ROLLING_AVG_WINDOW = 100

IMG_FORMAT = "pdf"
OUTPUT_DIR = f'output-{str(datetime.datetime.now()).replace(" ","-")}'

GENERALIZE_LEX_CONCEPTS = True
GENERALIZE_PERSONS = True

# Defaults for UserSettableParameters
# Independent variable
PROPORTION_L2 = 0.5
# Only crucial parameter
PRONOUN_DROP_PROB = 1.0
# Model expansions turned off by default
CAPACITY_L1 = 0  # (0=off)
CAPACITY_L2 = 0
# MIN_BOUNDARY_FEATURE_DIST = 0.0
# REDUCTION_HH = False
REDUCTION_PHONOTACTICS_L1 = False
REDUCTION_PHONOTACTICS_L2 = False
NEGATIVE_UPDATE = False
GEN_PRODUCTION_OLD_L1 = 0.0
GEN_PRODUCTION_OLD_L2 = 0.0
GEN_UPDATE_OLD_L1 = 0.0
GEN_UPDATE_OLD_L2 = 0.0
AFFIX_PRIOR_L1 = False
AFFIX_PRIOR_L2 = False
# Always affix setting simplifies model and disables suffix_prob
ALWAYS_AFFIX = True
SUFFIX_PROB = 0.5
# Settings to check which model results come from data artefacts
BALANCE_PREFIX_SUFFIX_VERBS = False
UNIQUE_AFFIX = False

SEND_EMPTY_IF_NONE = False
SYNTHETIC_FORMS = False


# For evaluation script (not browser visualization)
ITERATIONS = [5]
STEPS = [10000]


model_params = {
    "height": {"ui": HEIGHT, "script": HEIGHT},
    "width": {"ui": WIDTH, "script": WIDTH},
    "proportion_l2": {"ui": Slider("Proportion L2", PROPORTION_L2, 0.0, 1.0, 0.1), "script": PROPORTION_L2},
    "suffix_prob": {"ui": Slider("Suffix prob (intrans)", SUFFIX_PROB, 0.0, 1.0, 0.1), "script": SUFFIX_PROB},
    "capacity_l1": {"ui": Slider("Exemplar capacity L1", CAPACITY_L1, 0, 50, 1), "script": CAPACITY_L1},
    "capacity_l2": {"ui": Slider("Exemplar capacity L2", CAPACITY_L2, 0, 50, 1), "script": CAPACITY_L2},
    "pronoun_drop_prob": {"ui": Slider("Pronoun drop prob", PRONOUN_DROP_PROB, 0, 1, 0.1), "script": PRONOUN_DROP_PROB},
    # "min_boundary_feature_dist": {"ui": Slider("Min boundary feature dist",
    #                                                          MIN_BOUNDARY_FEATURE_DIST, 0, 10, 0.1), "script": MIN_BOUNDARY_FEATURE_DIST},
    # "reduction_hh": {"ui": Checkbox('Reduction H&H', value=REDUCTION_HH), "script": REDUCTION_HH},
    "reduction_phonotactics_l1": {"ui": Checkbox('Reduction phonotactics L1', value=REDUCTION_PHONOTACTICS_L1), "script": REDUCTION_PHONOTACTICS_L1},
    "reduction_phonotactics_l2": {"ui": Checkbox('Reduction phonotactics L2', value=REDUCTION_PHONOTACTICS_L2), "script": REDUCTION_PHONOTACTICS_L2},
    "negative_update": {"ui": Checkbox('Negative update', value=NEGATIVE_UPDATE), "script": NEGATIVE_UPDATE},
    "always_affix": {"ui": Checkbox('Always affix', value=ALWAYS_AFFIX), "script": ALWAYS_AFFIX},
    "balance_prefix_suffix_verbs": {"ui": Checkbox('Balance prefix/suffix', value=BALANCE_PREFIX_SUFFIX_VERBS), "script": BALANCE_PREFIX_SUFFIX_VERBS},
    "unique_affix": {"ui": Checkbox('Unique affix', value=UNIQUE_AFFIX), "script": UNIQUE_AFFIX},
    "send_empty_if_none": {"ui": Checkbox('Send empty if none', value=SEND_EMPTY_IF_NONE), "script": SEND_EMPTY_IF_NONE},
    "synthetic_forms": {"ui": Checkbox('Synthetic forms', value=SYNTHETIC_FORMS), "script": SYNTHETIC_FORMS},
    "gen_production_old_l1": {"ui": Slider("Generalize production L1 prob",
                                                          GEN_PRODUCTION_OLD_L1, 0, 1, 0.1), "script": GEN_PRODUCTION_OLD_L1},
    "gen_production_old_l2": {"ui": Slider("Generalize production L2 prob",
                                                          GEN_PRODUCTION_OLD_L2, 0, 1, 0.1), "script": GEN_PRODUCTION_OLD_L2},
    "gen_update_old_l1": {"ui": Slider("Generalize update L1 prob",
                                                      GEN_UPDATE_OLD_L1, 0, 1, 0.01), "script": GEN_UPDATE_OLD_L1},
    "gen_update_old_l2": {"ui": Slider("Generalize update L2 prob",
                                                      GEN_UPDATE_OLD_L2, 0, 1, 0.01), "script": GEN_UPDATE_OLD_L2},
    "affix_prior_l1": {"ui": Checkbox('Affix prior L1', value=AFFIX_PRIOR_L1), "script": AFFIX_PRIOR_L1},
    "affix_prior_l2": {"ui": Checkbox('Affix prior L2', value=AFFIX_PRIOR_L2), "script": AFFIX_PRIOR_L2},
    "browser_visualization": {"ui": True, "script": False},
}

model_params_ui = {k: v["ui"] for k, v in model_params.items()}
model_params_script = {k: v["script"] for k, v in model_params.items()}

evaluation_params = {
    "iterations": ITERATIONS,
    "steps": STEPS,
    "runlabel": "",
    "plot_from_raw": ""
}

bool_params = ["reduction_hh", "reduction_phonotactics_l1", "reduction_phonotactics_l2", "negative_update", "always_affix",
               "balance_prefix_suffix_verbs", "unique_affix", "send_empty_if_none",
               "synthetic_forms", "affix_prior_l1", "affix_prior_l2", "browser_visualization"]

string_params = ["runlabel", "plot_from_raw"]
