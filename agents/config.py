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

STATS_AFTER_STEPS = 20

# Defaults for UserSettableParameters
SUFFIX_PROB = 0.5
PROPORTION_L2 = 0.2
CAPACITY_L1 = 50
CAPACITY_L2 = 50
DROP_SUBJECT_PROB = 0.0
DROP_OBJECT_PROB = 0.0
MIN_BOUNDARY_FEATURE_DIST = 3.0 # 2.0

# For evaluation script (not browser visualization)
ITERATIONS = [3]
STEPS = [5000]

model_params = {
    "height": HEIGHT,
    "width": WIDTH,
    "proportion_l2": PROPORTION_L2,
    "suffix_prob": SUFFIX_PROB,
    "capacity_l1": CAPACITY_L1,
    "capacity_l2": CAPACITY_L2,
    "drop_subject_prob": DROP_SUBJECT_PROB,
    "drop_object_prob": DROP_OBJECT_PROB,
    "min_boundary_feature_dist": MIN_BOUNDARY_FEATURE_DIST
}

evaluation_params = {
    "iterations": ITERATIONS,
    "steps": STEPS
}