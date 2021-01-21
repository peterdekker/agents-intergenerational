import argparse
from mesa.batchrunner import BatchRunner
from constants import HEIGHT, WIDTH, PROPORTION_L2, SUFFIX_PROB, CAPACITY_L1, CAPACITY_L2, \
                        DROP_SUBJECT_PROB, DROP_OBJECT_PROB, MIN_BOUNDARY_FEATURE_DIST, STEPS, ITERATIONS
from Model import Model

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

stats = {"global_filled_prefix_l1": lambda m: m.global_filled_prefix_l1,
         "global_filled_suffix_l1": lambda m: m.global_filled_suffix_l1,
         "global_filled_prefix_l2": lambda m: m.global_filled_prefix_l2,
         "global_filled_suffix_l2": lambda m: m.global_filled_suffix_l2}


def params_print(params):
    return "".join([f"{k}: {v}   " for k, v in params.items()])


def evaluate_model(fixed_params, variable_params, iterations, steps):
    print(f"- Running batch: {iterations} iterations of {steps} steps")
    print(f"  Variable parameters: {params_print(variable_params)}")
    print(f"  Fixed parameters: {params_print(fixed_params)}")

    batch_run = BatchRunner(
        Model,
        variable_params,
        fixed_params,
        iterations=iterations,
        max_steps=steps,
        model_reporters=stats
    )

    batch_run.run_all()
    cols = list(variable_params.keys()) + list(stats.keys())
    run_data = batch_run.get_model_vars_dataframe()[cols]
    print()
    print(run_data)
    print("\n")
    run_data.to_csv(f"evaluation-{iterations}-{steps}.tsv", sep="\t")


def main():
    parser = argparse.ArgumentParser(description='Run agent model from terminal.')
    model_group = parser.add_argument_group('model', 'Model parameters')
    for param in model_params:
        model_group.add_argument(f"--{param}", nargs="+", type=float)
    evaluation_group = parser.add_argument_group('evaluation', 'Evaluation parameters')
    for param in evaluation_params:
        evaluation_group.add_argument(f"--{param}", nargs="+", type=int, default=evaluation_params[param])

    # Parse arguments
    args = vars(parser.parse_args())
    print(args)
    variable_params = {k: v for k, v in args.items() if k not in evaluation_params and v is not None}
    fixed_params = {k: v for k, v in model_params.items() if k not in variable_params}
    iterations = args["iterations"]
    steps = args["steps"]

    print(f"Evaluating iterations {iterations} and steps {steps}")
    for iterations_setting in iterations:
        for steps_setting in steps:
            evaluate_model(fixed_params, variable_params, iterations_setting, steps_setting)

if __name__ == "__main__":
    main()

