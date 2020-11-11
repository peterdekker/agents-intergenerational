import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

HEIGHT = 6
WIDTH = 6
MAX_RADIUS = max(HEIGHT, WIDTH)
N_AGENTS = HEIGHT*WIDTH
N_CONCEPTS = 10
N_FEATURES = 9
NOISE_RATE = 0.1
LEARNING_RATE = 0.5
SAMPLE = HEIGHT
HSAMPLE = int(SAMPLE/2)
DATA_FILE = "data/data.csv"
PERSONS = ['1sg', '2sg', '3sg', '1pl.incl', '1pl.excl', '2pl', '3pl']
RG = np.random.default_rng()
SUFFIX_PROB= 0.5
UPDATE_AMOUNT = 0.01
