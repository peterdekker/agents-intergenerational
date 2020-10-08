import numpy as np

HEIGHT = 10
WIDTH = 10
MAX_RADIUS = max(HEIGHT, WIDTH)
N_AGENTS = HEIGHT*WIDTH
N_CONCEPTS = 10
N_FEATURES = 9
NOISE_RATE = 0.1
LEARNING_RATE = 0.5
SAMPLE = N_AGENTS  # int(HEIGHT*WIDTH/2)
HSAMPLE = int(SAMPLE/2)
DATA_FILE = "data/data.csv"
PERSONS = ['1sg', '2sg', '3sg', '1pl.incl', '1pl.excl', '2pl', '3pl']
RG = np.random.default_rng()
SUFFIX_PROB= 0.5
