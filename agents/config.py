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
RARE_STATS_AFTER_STEPS = 500 * STATS_AFTER_STEPS

OUTPUT_DIR = f'output-{str(datetime.datetime.now()).replace(" ","-")}'

# Defaults for UserSettableParameters
PROPORTION_L2 = 0.5
SUFFIX_PROB = 0.5
CAPACITY_L1 = 50
CAPACITY_L2 = 50
DROP_SUBJECT_PROB = 0.5
MIN_BOUNDARY_FEATURE_DIST = 0.0
REDUCTION_HH = False
NEGATIVE_UPDATE = False
GENERALIZE_PRODUCTION_L1 = 0.0
GENERALIZE_PRODUCTION_L2 = 0.0
GENERALIZE_UPDATE_L1 = 0.0
GENERALIZE_UPDATE_L2 = 0.0
ALWAYS_AFFIX = True
BALANCE_PREFIX_SUFFIX_VERBS = False
FUZZY_MATCH_AFFIX = False

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
    "generalize_production_l1": GENERALIZE_PRODUCTION_L1,
    "generalize_production_l2": GENERALIZE_PRODUCTION_L2,
    "generalize_update_l1": GENERALIZE_UPDATE_L1,
    "generalize_update_l2": GENERALIZE_UPDATE_L2
}

evaluation_params = {
    "iterations": ITERATIONS,
    "steps": STEPS,
    "settings_graph": SETTINGS_GRAPH,
    "steps_graph": STEPS_GRAPH
}

bool_params = ["reduction_hh", "negative_update", "always_affix",
               "balance_prefix_suffix_verbs",  "fuzzy_match_affix", "settings_graph", "steps_graph"]
