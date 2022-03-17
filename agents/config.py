import numpy as np
import logging
import sys
import datetime
import os
from panphon.distance import Distance
from mesa.visualization.UserParam import UserSettableParameter

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

CURRENTDIR = os.path.dirname(os.path.realpath(__file__))
CLTS_ARCHIVE_PATH = os.path.join(CURRENTDIR, "2.1.0.tar.gz")
CLTS_ARCHIVE_URL = "https://github.com/cldf-clts/clts/archive/refs/tags/v2.1.0.tar.gz"
CLTS_PATH = os.path.join(CURRENTDIR, "clts-2.1.0")

HEIGHT = 6
WIDTH = 6
MAX_RADIUS = max(HEIGHT, WIDTH)

N_AGENTS = HEIGHT*WIDTH
SAMPLE = HEIGHT
DATA_FILE = "data/data.csv"
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
REDUCTION_PROSODY_L1 = False
REDUCTION_PROSODY_L2 = False
NEGATIVE_UPDATE = False
GENERALIZE_PRODUCTION_L1 = 0.0
GENERALIZE_PRODUCTION_L2 = 0.0
GENERALIZE_UPDATE_L1 = 0.0
GENERALIZE_UPDATE_L2 = 0.0
# Always affix setting simplifies model and disables suffix_prob
ALWAYS_AFFIX = True
SUFFIX_PROB = 0.5
# Settings to check which model results come from data artefacts
BALANCE_PREFIX_SUFFIX_VERBS = False
UNIQUE_AFFIX = False


# For evaluation script (not browser visualization)
ITERATIONS = [20]
STEPS = [8000]


model_params = {
    "height": {"ui": HEIGHT, "script": HEIGHT},
    "width": {"ui": WIDTH, "script": WIDTH},
    "proportion_l2": {"ui": UserSettableParameter("slider", "Proportion L2", PROPORTION_L2, 0.0, 1.0, 0.1), "script": PROPORTION_L2},
    "suffix_prob": {"ui": UserSettableParameter("slider", "Suffix prob (intrans)", SUFFIX_PROB, 0.0, 1.0, 0.1), "script": SUFFIX_PROB},
    "capacity_l1": {"ui": UserSettableParameter("slider", "Exemplar capacity L1", CAPACITY_L1, 0, 50, 1), "script": CAPACITY_L1},
    "capacity_l2": {"ui": UserSettableParameter("slider", "Exemplar capacity L2", CAPACITY_L2, 0, 50, 1), "script": CAPACITY_L2},
    "pronoun_drop_prob": {"ui": UserSettableParameter("slider", "Pronoun drop prob", PRONOUN_DROP_PROB, 0, 1, 0.1), "script": PRONOUN_DROP_PROB},
    # "min_boundary_feature_dist": {"ui": UserSettableParameter("slider", "Min boundary feature dist",
    #                                                          MIN_BOUNDARY_FEATURE_DIST, 0, 10, 0.1), "script": MIN_BOUNDARY_FEATURE_DIST},
    # "reduction_hh": {"ui": UserSettableParameter('checkbox', 'Reduction H&H', value=REDUCTION_HH), "script": REDUCTION_HH},
    "reduction_prosody_l1": {"ui": UserSettableParameter('checkbox', 'Reduction prosody L1', value=REDUCTION_PROSODY_L1), "script": REDUCTION_PROSODY_L1},
    "reduction_prosody_l2": {"ui": UserSettableParameter('checkbox', 'Reduction prosody L2', value=REDUCTION_PROSODY_L2), "script": REDUCTION_PROSODY_L2},
    "negative_update": {"ui": UserSettableParameter('checkbox', 'Negative update', value=NEGATIVE_UPDATE), "script": NEGATIVE_UPDATE},
    "always_affix": {"ui": UserSettableParameter('checkbox', 'Always affix', value=ALWAYS_AFFIX), "script": ALWAYS_AFFIX},
    "balance_prefix_suffix_verbs": {"ui": UserSettableParameter('checkbox', 'Balance prefix/suffix', value=BALANCE_PREFIX_SUFFIX_VERBS), "script": BALANCE_PREFIX_SUFFIX_VERBS},
    "unique_affix": {"ui": UserSettableParameter('checkbox', 'Unique affix', value=UNIQUE_AFFIX), "script": UNIQUE_AFFIX},
    "generalize_production_l1": {"ui": UserSettableParameter("slider", "Generalize production L1 prob",
                                                             GENERALIZE_PRODUCTION_L1, 0, 1, 0.1), "script": GENERALIZE_PRODUCTION_L1},
    "generalize_production_l2": {"ui": UserSettableParameter("slider", "Generalize production L2 prob",
                                                             GENERALIZE_PRODUCTION_L2, 0, 1, 0.1), "script": GENERALIZE_PRODUCTION_L2},
    "generalize_update_l1": {"ui": UserSettableParameter("slider", "Generalize update L1 prob",
                                                         GENERALIZE_UPDATE_L1, 0, 1, 0.01), "script": GENERALIZE_UPDATE_L1},
    "generalize_update_l2": {"ui": UserSettableParameter("slider", "Generalize update L2 prob",
                                                         GENERALIZE_UPDATE_L2, 0, 1, 0.01), "script": GENERALIZE_UPDATE_L2},
}

model_params_ui = {k: v["ui"] for k, v in model_params.items()}
model_params_script = {k: v["script"] for k, v in model_params.items()}

evaluation_params = {
    "iterations": ITERATIONS,
    "steps": STEPS,
    "runlabel": "",
    "plot_from_raw": ""
}

bool_params = ["reduction_hh", "reduction_prosody_l1", "reduction_prosody_l2", "negative_update", "always_affix",
               "balance_prefix_suffix_verbs", "unique_affix"]

string_params = ["runlabel", "plot_from_raw"]
