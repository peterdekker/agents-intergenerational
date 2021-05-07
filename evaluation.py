import argparse
from mesa.batchrunner import BatchRunner

from agents.model import Model
from agents.config import model_params, evaluation_params, bool_params

import matplotlib.pyplot as plt
import numpy as np

import textwrap

stats = {"global_filled_prefix_l1": lambda m: m.global_filled_prefix_l1,
         "global_filled_suffix_l1": lambda m: m.global_filled_suffix_l1,
         "global_filled_prefix_l2": lambda m: m.global_filled_prefix_l2,
         "global_filled_suffix_l2": lambda m: m.global_filled_suffix_l2}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    return run_data


def create_graph(run_data, fixed_params, variable_param):
    run_data_means = run_data.groupby(variable_param).mean()
    stats_cols = run_data_means.columns  # Statistics: suffix L1, etc.
    labels = run_data_means.index  # variable values
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    rects = {}
    for i, stats_col in enumerate(stats_cols):
        rects[stats_col] = ax.bar(x+i*width, run_data_means[stats_col],
                                  width=width, edgecolor="white", label=stats_col.strip("global_filled_"))
        #ax.bar_label(rects[stats_col], padding=3)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% paradigm cells filled')
    ax.set_title(variable_param)
    ax.set_xticks(x+1.5*width)
    ax.set_xticklabels(labels)
    ax.legend()
    # fig.tight_layout()
    graphtext = textwrap.fill(params_print(fixed_params), width=100)
    plt.subplots_adjust(bottom=0.2)
    plt.figtext(0.05, 0.03, graphtext, fontsize=8, ha="left")
    plt.savefig(f"{variable_param}.png")  # bbox_inches="tight"


def main():
    parser = argparse.ArgumentParser(description='Run agent model from terminal.')
    model_group = parser.add_argument_group('model', 'Model parameters')
    for param in model_params:
        model_group.add_argument(f"--{param}", nargs="+",
                                 type=str2bool if param in bool_params else float)
    evaluation_group = parser.add_argument_group('evaluation', 'Evaluation parameters')
    for param in evaluation_params:
        if param in bool_params:
            evaluation_group.add_argument(f'--{param}', action='store_true')
        else:
            evaluation_group.add_argument(f"--{param}", nargs="+", type=int, default=evaluation_params[param])

    # Parse arguments
    args = vars(parser.parse_args())
    variable_params = {k: v for k, v in args.items() if k in model_params and v is not None}
    iterations = args["iterations"]
    steps = args["steps"]
    compare_graph = args["compare_graph"]
    # if compare_graph and len(variable_params) != 1:
    #     raise ValueError(
    #         "With option --compare_graph, please supply EXACTLY ONE model variable to evaluate in graph.")

    print(f"Evaluating iterations {iterations} and steps {steps}")
    if compare_graph:
        # Try variable parameters one by one, while keeping all of the other parameters fixed
        for var_param, var_param_setting in variable_params.items():
            for iterations_setting in iterations:
                for steps_setting in steps:
                    fixed_params_other = {k: v for k, v in model_params.items() if k != var_param}
                    run_data = evaluate_model(fixed_params_other, {var_param: var_param_setting},
                                              iterations_setting, steps_setting)
                    create_graph(run_data, fixed_params_other, var_param)
    else:
        # Evaluate all combinations of variable parameters
        # Only params not changed by user are fixed
        fixed_params = {k: v for k, v in model_params.items() if k not in variable_params}
        for iterations_setting in iterations:
            for steps_setting in steps:
                run_data = evaluate_model(fixed_params, variable_params, iterations_setting, steps_setting)


if __name__ == "__main__":
    main()
