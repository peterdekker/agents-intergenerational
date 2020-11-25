import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

HEIGHT = 6
WIDTH = 6
MAX_RADIUS = max(HEIGHT, WIDTH)

N_AGENTS = HEIGHT*WIDTH
SAMPLE = HEIGHT
HSAMPLE = int(SAMPLE/2)
DATA_FILE = "data/data.csv"
PERSONS = ['1sg', '2sg', '3sg', '1pl.incl', '1pl.excl', '2pl', '3pl']
RG = np.random.default_rng()
SUFFIX_PROB = 0.5

# Defaults for UserSettableParameters
PROPORTION_L2 = 0.2
CAPACITY_L1 = 100
CAPACITY_L2 = 100
