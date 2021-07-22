import argparse
from mesa.batchrunner import BatchRunner

from agents.model import Model
from agents import misc
from agents.config import model_params, evaluation_params, bool_params, OUTPUT_DIR, IMG_FORMAT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import textwrap
import os

stats_internal = {"prop_internal_prefix_l1": lambda m: m.prop_internal_prefix_l1,
                  "prop_internal_suffix_l1": lambda m: m.prop_internal_suffix_l1,
                  "prop_internal_prefix_l2": lambda m: m.prop_internal_prefix_l2,
                  "prop_internal_suffix_l2": lambda m: m.prop_internal_suffix_l2}


stats_communicated = {"prop_communicated_prefix_l1": lambda m: m.prop_communicated_prefix_l1,
                      "prop_communicated_suffix_l1": lambda m: m.prop_communicated_suffix_l1,
                      "prop_communicated_prefix_l2": lambda m: m.prop_communicated_prefix_l2,
                      "prop_communicated_suffix_l2": lambda m: m.prop_communicated_suffix_l2}

stats = {**stats_internal, **stats_communicated}


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


def evaluate_model(fixed_params, variable_params, iterations, steps, output_dir):
    print(f"- Running batch: {iterations} iterations of {steps} steps")
    print(f"  Variable parameters: {params_print(variable_params)}")
    print(f"  Fixed parameters: {params_print(fixed_params)}")

    batch_run = BatchRunner(
        Model,
        variable_params,
        fixed_params,
        iterations=iterations,
        max_steps=steps,
        model_reporters={**stats, **{"datacollector": lambda m: m.datacollector}}
    )

    batch_run.run_all()

    # cols_internal = list(variable_params.keys()) + list(stats_internal.keys())
    # run_data_internal = batch_run.get_model_vars_dataframe()[cols_internal]
    # run_data_internal.to_csv(f"evaluation-internal-{iterations}-{steps}.tsv", sep="\t")

    # cols_communicated = list(variable_params.keys()) + list(stats_communicated.keys())
    # run_data_communicated = batch_run.get_model_vars_dataframe()[cols_communicated]
    # run_data_communicated.to_csv(f"evaluation-communicated-{iterations}-{steps}.tsv", sep="\t")

    run_data = batch_run.get_model_vars_dataframe()
    run_data.to_csv(os.path.join(output_dir, f"evaluation-{iterations}-{steps}.tsv"), sep="\t")

    # print()
    # print(run_data)
    # print("\n")
    return run_data


def create_graph_course(run_data, fixed_params, variable_param, variable_param_settings, mode, stats, stat, output_dir):
    course_df = get_course_df(run_data, variable_param, variable_param_settings, stats)
    plot_graph_course(course_df, fixed_params, variable_param,
                      variable_param_settings, stat, mode, output_dir)


def get_course_df(run_data, variable_param, variable_param_settings, stats):
    multi_index = pd.MultiIndex.from_product([variable_param_settings, stats])
    course_df = pd.DataFrame(columns=multi_index)
    for param_setting, group in run_data.groupby(variable_param):
        iteration_dfs = []
        for i, row in group.iterrows():
            iteration_df = row["datacollector"].get_model_vars_dataframe()[stats]
            iteration_dfs.append(iteration_df)
        iteration_dfs_concat = pd.concat(iteration_dfs)
        # TODO: spread instead of mean
        combined = iteration_dfs_concat.groupby(iteration_dfs_concat.index).mean()  # group by index
        for stat_col in combined:
            course_df[param_setting, stat_col] = combined[stat_col]
    # Drop first row of course df, because this is logging artefact
    course_df = course_df.iloc[1:, :]
    return course_df
    # TODO: possibly function intersection here later

# Needed for function intersection
# def pos_under_val(arr, val):
#     previous_i = 0
#     for i in reversed(range(len(arr))):
#         if arr[i] > val:
#             return previous_i
#         previous_i = i
#     return previous_i


def plot_graph_course(course_df, fixed_params, variable_param, variable_param_settings, stat, mode, output_dir):
    fig, ax = plt.subplots()
    steps_ix = course_df.index
    for param_setting in variable_param_settings:
        ax.plot(steps_ix, course_df[param_setting, stat],
                label=f"{variable_param}={param_setting}", linewidth=1.0)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if mode == "internal":
        ax.set_ylabel('proportion paradigm cells filled')
    elif mode == "communicated":
        ax.set_ylabel('proportion non-empty utterances')
    ax.set_title(f"{variable_param} ({mode})")
    # ax.set_xticks(x+1.5*width)
    # ax.set_xticklabels(labels)
    ax.legend()
    # fig.tight_layout()
    graphtext = textwrap.fill(params_print(fixed_params), width=100)
    plt.subplots_adjust(bottom=0.2)
    plt.figtext(0.05, 0.03, graphtext, fontsize=8, ha="left")
    # bbox_inches="tight"
    plt.savefig(os.path.join(output_dir, f"{variable_param}-{mode}-course.{IMG_FORMAT}"), format=IMG_FORMAT)


def create_graph_end(run_data, fixed_params, variable_param, variable_param_settings, mode, stats, output_dir):
    course_df = get_course_df(run_data, variable_param, variable_param_settings, stats)
    print(course_df.tail(50).mean())

    run_data_means = run_data.groupby(variable_param).mean(numeric_only=True)
    print(run_data_means)
    labels = run_data_means.index  # variable values
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    rects = {}
    for i, stat in enumerate(stats):
        stat_label = stat.replace(
            "prop_internal_", "") if mode == "internal" else stat.replace("prop_communicated_", "")
        rects[stat] = ax.bar(x+i*width, run_data_means[stat],
                             width=width, edgecolor="white", label=stat_label)
        # ax.bar_label(rects[stats_col], padding=3)
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    if mode == "internal":
        ax.set_ylabel('proportion paradigm cells filled')
    elif mode == "communicated":
        ax.set_ylabel('proportion utterances non-empty')
    ax.set_title(f"{variable_param} ({mode})")
    ax.set_xticks(x+1.5*width)
    ax.set_xticklabels(labels)
    ax.legend()
    # fig.tight_layout()
    graphtext = textwrap.fill(params_print(fixed_params), width=100)
    plt.subplots_adjust(bottom=0.2)
    plt.figtext(0.05, 0.03, graphtext, fontsize=8, ha="left")
    # bbox_inches="tight"
    plt.savefig(os.path.join(output_dir, f"{variable_param}-{mode}-end.{IMG_FORMAT}"), format=IMG_FORMAT)


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
    settings_graph = args["settings_graph"]
    steps_graph = args["steps_graph"]

    if settings_graph:
        if (len(iterations) != 1 or len(steps) != 1):
            raise ValueError(
                "With option --settings_graph, only supply one iterations setting and one steps setting (or none for default).")
        if (len(variable_params) == 0):
            raise ValueError(
                "With option --settings_graph, please supply one or more variable parameters.")

    if steps_graph:
        if (len(variable_params) != 0 or len(iterations) != 1):
            raise ValueError(
                "With option --steps_graph, please do not supply variable parameters. Also, supply only one iterations setting, or none for default.")
        if (len(steps) == 0):
            # This does not happen easily in practice, because of default value
            raise ValueError(
                "With option --steps_graph, please supply one or multiple steps settings")

    print(f"Evaluating iterations {iterations} and steps {steps}")

    misc.create_output_dir(OUTPUT_DIR)

    if settings_graph:
        # Try variable parameters one by one, while keeping all of the other parameters fixed
        for var_param, var_param_settings in variable_params.items():
            assert len(iterations) == 1
            assert len(steps) == 1
            iterations_setting = iterations[0]
            steps_setting = steps[0]
            fixed_params = {k: v for k, v in model_params.items() if k != var_param}
            fixed_params_print = {**fixed_params, **
                                  {"iterations": iterations_setting, "steps": steps_setting}}
            run_data = evaluate_model(fixed_params, {var_param: var_param_settings},
                                      iterations_setting, steps_setting, output_dir=OUTPUT_DIR)
            # create_graph_end(run_data, fixed_params_print, var_param,
            #                        mode="internal", stats=stats_internal, output_dir=OUTPUT_DIR)
            create_graph_end(run_data, fixed_params_print, var_param, var_param_settings,
                             mode="communicated", stats=stats_communicated, output_dir=OUTPUT_DIR)
            # create_graph_course(run_data, fixed_params_print, var_param,
            #                     mode="internal", stat="prop_internal_suffix_l1", output_dir=OUTPUT_DIR)
            create_graph_course(run_data, fixed_params_print, var_param, var_param_settings,
                                mode="communicated", stats=stats_communicated.keys(),
                                stat="prop_communicated_suffix_l1", output_dir=OUTPUT_DIR)
    elif steps_graph:
        # No variable parameters are used and no iterations are used. Only evaluate
        run_data_list = []
        assert len(iterations) == 1
        iterations_setting = iterations[0]
        for steps_setting in steps:
            fixed_params = model_params
            run_data = evaluate_model(fixed_params, {},
                                      iterations_setting, steps_setting, OUTPUT_DIR)
            run_data["steps"] = steps_setting
            run_data_list.append(run_data)
        combined_run_data = pd.concat(run_data_list, ignore_index=True)
        fixed_params_print = {**fixed_params, **{"iterations": iterations_setting}}
        # create_graph_end(combined_run_data, fixed_params_print,
        #                        "steps", mode="internal", stats=stats_internal)
        create_graph_end(combined_run_data, fixed_params_print, "steps",
                         mode="communicated", stats=stats_communicated)
    else:
        # Evaluate all combinations of variable parameters
        # Only params not changed by user are fixed
        fixed_params = {k: v for k, v in model_params.items() if k not in variable_params}
        for iterations_setting in iterations:
            for steps_setting in steps:
                run_data = evaluate_model(fixed_params, variable_params,
                                          iterations_setting, steps_setting, OUTPUT_DIR)


if __name__ == "__main__":
    main()
