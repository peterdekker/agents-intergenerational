import numpy as np
import logging
import sys
import datetime
from panphon.distance import Distance

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

HEIGHT = 6
WIDTH = 6
MAX_RADIUS = max(HEIGHT, WIDTH)

N_AGENTS = HEIGHT*WIDTH
SAMPLE = HEIGHT
DATA_FILE = "data/data.csv"
PERSONS = ['1sg', '2sg', '3sg', '1pl.incl', '1pl.excl', '2pl', '3pl']

dst = Distance()
RG = np.random.default_rng()

STATS_AFTER_STEPS = 1
RARE_STATS_AFTER_STEPS = 1000

# Last n number of steps to take into account, when calculating communicated stat
#LAST_N_STEPS_COMMUNICATED = 1
# Last n number of steps to take into account, when calculating end state graph
LAST_N_STEPS_END_GRAPH = 500

IMG_FORMAT = "png"
OUTPUT_DIR = f'output-{str(datetime.datetime.now()).replace(" ","-")}'

# Defaults for UserSettableParameters
# Independent variable
PROPORTION_L2 = 0.5
# Only crucial parameter
DROP_SUBJECT_PROB = 0.5
# Model expansions turned off by default
CAPACITY_L1 = 50
CAPACITY_L2 = 50
MIN_BOUNDARY_FEATURE_DIST = 0.0
REDUCTION_HH = False
NEGATIVE_UPDATE = False
GENERALIZE_PRODUCTION_L1 = 0.0
GENERALIZE_PRODUCTION_L2 = 0.0
GENERALIZE_UPDATE_L1 = 0.0
GENERALIZE_UPDATE_L2 = 0.0
FUZZY_MATCH_AFFIX = False
# Always affix setting simplifies model and disables suffix_prob
ALWAYS_AFFIX = True
SUFFIX_PROB = 0.5
# Settings to check which model results come from data artefacts
BALANCE_PREFIX_SUFFIX_VERBS = False
UNIQUE_AFFIX = False


# For evaluation script (not browser visualization)
ITERATIONS = [3]
STEPS = [5000]
SETTINGS_GRAPH = False
STEPS_GRAPH = False

model_params = {
    "height": HEIGHT,
    "width": WIDTH,
    "proportion_l2": PROPORTION_L2,
    "suffix_prob": SUFFIX_PROB,
    "capacity_l1": CAPACITY_L1,
    "capacity_l2": CAPACITY_L2,
    "drop_subject_prob": DROP_SUBJECT_PROB,
    "min_boundary_feature_dist": MIN_BOUNDARY_FEATURE_DIST,
    "reduction_hh": REDUCTION_HH,
    "negative_update": NEGATIVE_UPDATE,
    "always_affix": ALWAYS_AFFIX,
    "fuzzy_match_affix": FUZZY_MATCH_AFFIX,
    "balance_prefix_suffix_verbs": BALANCE_PREFIX_SUFFIX_VERBS,
    "unique_affix": UNIQUE_AFFIX,
    "generalize_production_l1": GENERALIZE_PRODUCTION_L1,
    "generalize_production_l2": GENERALIZE_PRODUCTION_L2,
    "generalize_update_l1": GENERALIZE_UPDATE_L1,
    "generalize_update_l2": GENERALIZE_UPDATE_L2
}

evaluation_params = {
    "iterations": ITERATIONS,
    "steps": STEPS,
    "settings_graph": SETTINGS_GRAPH,
    "steps_graph": STEPS_GRAPH,
    "runlabel": ""
}

bool_params = ["reduction_hh", "negative_update", "always_affix",
               "balance_prefix_suffix_verbs", "unique_affix", "fuzzy_match_affix", "settings_graph", "steps_graph"]

string_params = ["runlabel"]
