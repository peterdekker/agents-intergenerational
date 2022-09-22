import numpy as np
import logging
import sys
import datetime
import os
from panphon.distance import Distance
from mesa.visualization.UserParam import Slider, Checkbox

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

CURRENTDIR = os.path.dirname(os.path.realpath(__file__))
CLTS_ARCHIVE_PATH = os.path.join(CURRENTDIR, "2.1.0.tar.gz")
CLTS_ARCHIVE_URL = "https://github.com/cldf-clts/clts/archive/refs/tags/v2.1.0.tar.gz"
CLTS_PATH = os.path.join(CURRENTDIR, "clts-2.1.0")

N_AGENTS = 10
DATA_FILE = "data/data-sample3.csv"
# DATA_FILE_SYNTHETIC = "data/data-syntheticforms-sample.csv"
PERSONS = ['1sg', '2sg', '3sg', '1pl.incl', '1pl.excl', '2pl', '3pl']

RG = np.random.default_rng()

ROLLING_AVG_WINDOW = 100

IMG_FORMAT = "pdf"
OUTPUT_DIR = f'output-{str(datetime.datetime.now()).replace(" ","-")}'


# Defaults for UserSettableParameters
# Independent variable
PROPORTION_L2 = 0.5
REDUCTION_PHONOTACTICS_L1 = False
REDUCTION_PHONOTACTICS_L2 = False
AFFIX_PRIOR_L1 = False
AFFIX_PRIOR_L2 = False
GENERALIZE_PERSONS = True
GENERALIZE_LEX_CONCEPTS = True
ALPHA_L1 = 1 
ALPHA_L2 = 1 #1000


ITERATIONS = [5]
STEPS = [10]
N_INTERACTIONS_PER_STEP = 10


model_params = {
    "n_agents": {"ui": N_AGENTS, "script": N_AGENTS},
    "proportion_l2": {"ui": Slider("Proportion L2", PROPORTION_L2, 0.0, 1.0, 0.1), "script": PROPORTION_L2},
    "reduction_phonotactics_l1": {"ui": Checkbox('Reduction phonotactics L1', value=REDUCTION_PHONOTACTICS_L1), "script": REDUCTION_PHONOTACTICS_L1},
    "reduction_phonotactics_l2": {"ui": Checkbox('Reduction phonotactics L2', value=REDUCTION_PHONOTACTICS_L2), "script": REDUCTION_PHONOTACTICS_L2},
    "affix_prior_l1": {"ui": Checkbox('Affix prior L1', value=AFFIX_PRIOR_L1), "script": AFFIX_PRIOR_L1},
    "affix_prior_l2": {"ui": Checkbox('Affix prior L2', value=AFFIX_PRIOR_L2), "script": AFFIX_PRIOR_L2},
    "alpha_l1": {"ui": Slider("alpha L1", ALPHA_L1, 0, 10000, 100), "script": ALPHA_L1},
    "alpha_l2": {"ui": Slider("alpha L2", ALPHA_L2, 0, 10000, 100), "script": ALPHA_L2},
}

model_params_ui = {k: v["ui"] for k, v in model_params.items()}
model_params_script = {k: v["script"] for k, v in model_params.items()}

evaluation_params = {
    "iterations": ITERATIONS,
    "steps": STEPS,
    "n_interactions_per_step": N_INTERACTIONS_PER_STEP,
    "runlabel": "",
    "plot_from_raw": ""
}

bool_params = ["reduction_phonotactics_l1", "reduction_phonotactics_l2", "affix_prior_l1", "affix_prior_l2"]

string_params = ["runlabel", "plot_from_raw"]
