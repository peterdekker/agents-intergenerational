from mesa.batchrunner import BatchRunner
from constants import HEIGHT, WIDTH, PROPORTION_L2, SUFFIX_PROB, CAPACITY_L1, CAPACITY_L2, \
                        DROP_SUBJECT_PROB, DROP_OBJECT_PROB, MIN_BOUNDARY_FEATURE_DIST
from Model import Model

fixed_params = {
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

variable_params = {"capacity_l1": [0,1,2,5,10], "capacity_l2": [0,1,2,5,10], "proportion_l2":[0.0, 0.2, 0.5]}
for k in variable_params:
    del fixed_params[k]

stats = {"global_filled_prefix_l1": lambda m: m.global_filled_prefix_l1,
         "global_filled_suffix_l1": lambda m: m.global_filled_suffix_l1,
         "global_filled_prefix_l2": lambda m: m.global_filled_prefix_l2,
         "global_filled_suffix_l2": lambda m: m.global_filled_suffix_l2}

batch_run = BatchRunner(
    Model,
    variable_params,
    fixed_params,
    iterations=3,
    max_steps=5000,
    model_reporters=stats
)

batch_run.run_all()
cols = list(variable_params.keys()) + list(stats.keys())
run_data = batch_run.get_model_vars_dataframe()[cols]
print(run_data)
run_data.to_csv("evaluation.tsv", sep="\t")