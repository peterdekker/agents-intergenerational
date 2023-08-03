import numpy as np
import logging
import sys
import datetime
import os
#from mesa.visualization.UserParam import Slider, Checkbox

# Parameters not settable by user
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

CURRENTDIR = os.path.dirname(os.path.realpath(__file__))
CLTS_ARCHIVE_PATH = os.path.join(CURRENTDIR, "2.1.0.tar.gz")
CLTS_ARCHIVE_URL = "https://github.com/cldf-clts/clts/archive/refs/tags/v2.1.0.tar.gz"
CLTS_PATH = os.path.join(CURRENTDIR, "clts-2.1.0")

N_AGENTS = 10
DATA_FILE = "data/data.csv"
# DATA_FILE_SYNTHETIC = "data/data-syntheticforms-sample.csv"
PERSONS = ['1sg', '2sg', '3sg', '1pl.incl', '1pl.excl', '2pl', '3pl']

RG = np.random.default_rng()

ROLLING_AVG_WINDOW = 100

GENERALIZE_PERSONS = False
GENERALIZE_LEX_CONCEPTS = True
GENERALIZE_PREFIX_SUFFIX = False

IMG_FORMATS = ["pdf", "png"]
OUTPUT_DIR = f'output-{str(datetime.datetime.now()).replace(" ","-").replace(":",".")}'

ENABLE_MULTITHREADING = False


# User settable parameteres
# Model parameters
PROPORTION_L2 = 0.5
PHONOTACTIC_REDUCTION_L1 = True
PHONOTACTIC_REDUCTION_L2 = True
PHONOTACTIC_REDUCTION_PROB = 1.0
PHONOTACTIC_REDUCTION_DROP_BORDER_PHONEME = False
AFFIX_PRIOR_COMBINED_L1 = False
AFFIX_PRIOR_COMBINED_L2 = False
AFFIX_PRIOR_L1 = True
AFFIX_PRIOR_L2 = True
AFFIX_PRIOR_PROB = 0.5
ALPHA_L1 = 1
ALPHA_L2 = 1  # 1000
INTERACTION_L1 = False
INTERACTION_L1_SHIELD_INITIALIZATION = 10
INTERACTIONS_PER_GENERATION = 10000

# Evaluateion parameters
ITERATIONS = 10
GENERATIONS = 50

# Backup with UI elements
# model_params = {
#     "n_agents": {"ui": N_AGENTS, "script": N_AGENTS},
#     "proportion_l2": {"ui": Slider("proportion_l2", PROPORTION_L2, 0.0, 1.0, 0.1), "script": PROPORTION_L2},
#     "phonotactic_reduction_l1": {"ui": Checkbox('Reduction phonotactics L1', value=PHONOTACTIC_REDUCTION_L1), "script": PHONOTACTIC_REDUCTION_L1},
#     "phonotactic_reduction_l2": {"ui": Checkbox('Reduction phonotactics L2', value=PHONOTACTIC_REDUCTION_L2), "script": PHONOTACTIC_REDUCTION_L2},
#     "affix_prior_l1": {"ui": Checkbox('Affix prior L1', value=AFFIX_PRIOR_L1), "script": AFFIX_PRIOR_L1},
#     "affix_prior_l2": {"ui": Checkbox('Affix prior L2', value=AFFIX_PRIOR_L2), "script": AFFIX_PRIOR_L2},
#     "alpha_l1": {"ui": Slider("alpha L1", ALPHA_L1, 0, 10000, 100), "script": ALPHA_L1},
#     "alpha_l2": {"ui": Slider("alpha L2", ALPHA_L2, 0, 10000, 100), "script": ALPHA_L2},
#     "interaction_l1": {"ui": Checkbox('Interaction L1', value=INTERACTION_L1), "script": INTERACTION_L1},
#     "interaction_l1_shield_initialization": {"ui": Slider("Interaction L1 shield initialization", INTERACTION_L1_SHIELD_INITIALIZATION, 0, 1000, 10), "script": INTERACTION_L1_SHIELD_INITIALIZATION},
#     "interactions_per_generation": {"script": INTERACTIONS_PER_GENERATION},
#     "generations": {"script": GENERATIONS},
# }

model_params = {
    "n_agents": {"ui": N_AGENTS, "script": N_AGENTS},
    "proportion_l2": {"script": PROPORTION_L2},
    "phonotactic_reduction_l1": {"script": PHONOTACTIC_REDUCTION_L1},
    "phonotactic_reduction_l2": {"script": PHONOTACTIC_REDUCTION_L2},
    "phonotactic_reduction_prob": {"script": PHONOTACTIC_REDUCTION_PROB},
    "phonotactic_reduction_drop_border_phoneme": {"script": PHONOTACTIC_REDUCTION_DROP_BORDER_PHONEME},
    "affix_prior_combined_l1": {"script": AFFIX_PRIOR_COMBINED_L1},
    "affix_prior_combined_l2": {"script": AFFIX_PRIOR_COMBINED_L2},
    "affix_prior_l1": {"script": AFFIX_PRIOR_L1},
    "affix_prior_l2": {"script": AFFIX_PRIOR_L2},
    "affix_prior_prob": {"script": AFFIX_PRIOR_PROB},
    "alpha_l1": {"script": ALPHA_L1},
    "alpha_l2": {"script": ALPHA_L2},
    "interaction_l1": {"script": INTERACTION_L1},
    "interaction_l1_shield_initialization": {"script": INTERACTION_L1_SHIELD_INITIALIZATION},
    "interactions_per_generation": {"script": INTERACTIONS_PER_GENERATION},
}

# model_params_ui = {k: v["ui"] for k, v in model_params.items()}
model_params_script = {k: v["script"] for k, v in model_params.items()}

evaluation_params = {
    "iterations": {"script": ITERATIONS},
    "runlabel": {"script": ""},
    "plot_from_raw": {"script": ""},
    "evaluate_prop_l2": {"script": False},
    "evaluate_param": {"script": False},
    "evaluate_params_heatmap": {"script": False},
    "generations": {"script": GENERATIONS},
}

eval_params_script = {k: v["script"] for k, v in evaluation_params.items()}

bool_params = ["phonotactic_reduction_l1", "phonotactic_reduction_l2", "phonotactic_reduction_drop_border_phoneme",
               "affix_prior_combined_l1", "affix_prior_combined_l2", "affix_prior_l1", "affix_prior_l2", "interaction_l1", "evaluate_prop_l2", "evaluate_param", "evaluate_params_heatmap"]

string_params = ["runlabel", "plot_from_raw"]
