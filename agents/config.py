import numpy as np
import logging
import sys
from panphon.distance import Distance

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

HEIGHT = 6
WIDTH = 6
MAX_RADIUS = max(HEIGHT, WIDTH)

N_AGENTS = HEIGHT*WIDTH
SAMPLE = HEIGHT
HSAMPLE = int(SAMPLE/2)
DATA_FILE = "data/data.csv"
PERSONS = ['1sg', '2sg', '3sg', '1pl.incl', '1pl.excl', '2pl', '3pl']

dst = Distance()
RG = np.random.default_rng()

STATS_AFTER_STEPS = 50
RARE_STATS_AFTER_STEPS = 2 * STATS_AFTER_STEPS  # Should always be multiple of STATS_AFTER_STEPS

# Defaults for UserSettableParameters
PROPORTION_L2 = 0.7
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

# For evaluation script (not browser visualization)
ITERATIONS = [3]
STEPS = [5000]
COMPARE_GRAPH = False

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
    "generalize_production_l1": GENERALIZE_PRODUCTION_L1,
    "generalize_production_l2": GENERALIZE_PRODUCTION_L2,
    "generalize_update_l1": GENERALIZE_UPDATE_L1,
    "generalize_update_l2": GENERALIZE_UPDATE_L2
}

evaluation_params = {
    "iterations": ITERATIONS,
    "steps": STEPS,
    "compare_graph": COMPARE_GRAPH
}

bool_params = ["reduction_hh", "negative_update", "compare_graph"]
