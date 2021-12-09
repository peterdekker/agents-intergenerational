import argparse
from mesa.batchrunner import BatchRunner

from agents.model import Model
from agents import misc
from agents.config import model_params_script, evaluation_params, bool_params, string_params, OUTPUT_DIR, IMG_FORMAT, ROLLING_AVG_WINDOW

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import textwrap
import os

# stats_internal = {"prop_internal_prefix_l1": lambda m: m.prop_internal_prefix_l1,
#                   "prop_internal_suffix_l1": lambda m: m.prop_internal_suffix_l1,
#                   "prop_internal_prefix_l2": lambda m: m.prop_internal_prefix_l2,
#                   "prop_internal_suffix_l2": lambda m: m.prop_internal_suffix_l2}


# stats_communicated = {"prop_communicated_prefix_l1": lambda m: m.prop_communicated_prefix_l1,
#                       "prop_communicated_suffix_l1": lambda m: m.prop_communicated_suffix_l1,
#                       "prop_communicated_prefix_l2": lambda m: m.prop_communicated_prefix_l2,
#                       "prop_communicated_suffix_l2": lambda m: m.prop_communicated_suffix_l2}

stats_internal = ["prop_internal_prefix_l1", "prop_internal_suffix_l1", "prop_internal_prefix_l2", "prop_internal_suffix_l2"]
stats_communicated = ["prop_communicated_prefix_l1", "prop_communicated_suffix_l1", "prop_communicated_prefix_l2", "prop_communicated_suffix_l2"]

#stats = {**stats_internal, **stats_communicated}


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
        model_reporters={"datacollector": lambda m: m.datacollector}
    )

    batch_run.run_all()

    run_data = batch_run.get_model_vars_dataframe()
    return run_data


def create_graph_course(run_data, fixed_params, variable_param, variable_param_settings, mode, stats, stat, output_dir):
    course_df = get_course_df(run_data, variable_param, variable_param_settings, stats)
    plot_graph_course(course_df, fixed_params, variable_param,
                      variable_param_settings, stat, mode, output_dir)

def create_graph_end(run_data, fixed_params, variable_param, variable_param_settings, mode, stats, output_dir):
    course_df = get_course_df(run_data, variable_param, variable_param_settings, stats)
    course_tail_avg = course_df.tail(LAST_N_STEPS_END_GRAPH).mean()
    # run_data_means = run_data.groupby(variable_param).mean(numeric_only=True)
    labels = variable_param_settings  # run_data_means.index  # variable values
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    rects = {}
    colors = ["deepskyblue", "royalblue", "orange", "darkgoldenrod"]
    for i, stat in enumerate(stats):
        stat_label = stat.replace(
            "prop_internal_", "") if mode == "internal" else stat.replace("prop_communicated_", "")
        rects[stat] = ax.bar(x+i*width, course_tail_avg[:,stat],
                             width=width, edgecolor="white", label=stat_label, color=colors[i])
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    if mode == "internal":
        ax.set_ylabel('proportion paradigm cells filled')
    elif mode == "communicated":
        ax.set_ylabel('proportion utterances non-empty')
    ax.set_xlabel(variable_param)
    ax.set_title(f"{variable_param} ({mode})")
    ax.set_xticks(x+1.5*width)
    ax.set_xticklabels(labels)
    ax.legend()
    graphtext = textwrap.fill(params_print(fixed_params), width=100)
    plt.subplots_adjust(bottom=0.25)
    plt.figtext(0.05, 0.03, graphtext, fontsize=8, ha="left")
    plt.savefig(os.path.join(output_dir, f"{variable_param}-{mode}-end.{IMG_FORMAT}"), format=IMG_FORMAT)


def get_course_df(run_data, variable_param, variable_param_settings, stats):
    multi_index = pd.MultiIndex.from_product([variable_param_settings, stats])
    course_df = pd.DataFrame(columns=multi_index)
    for param_setting, group in run_data.groupby(variable_param):
        iteration_dfs = []
        for i, row in group.iterrows():
            iteration_df = row["datacollector"].get_model_vars_dataframe()[stats]
            iteration_dfs.append(iteration_df)
        iteration_dfs_concat = pd.concat(iteration_dfs)
        # Group all iterations together for this index  # TODO: spread?
        combined = iteration_dfs_concat.groupby(iteration_dfs_concat.index).mean()
        for stat_col in combined:
            course_df[param_setting, stat_col] = combined[stat_col]
    # Drop first row of course df, because this is logging artefact
    course_df = course_df.iloc[1:, :]
    return course_df
    # TODO: possibly function intersection here later


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
    ax.set_xlabel(variable_param)
    ax.set_title(f"{variable_param} ({mode})")
    # ax.set_xticks(x+1.5*width)
    # ax.set_xticklabels(labels)
    ax.legend()
    # fig.tight_layout()
    graphtext = textwrap.fill(params_print(fixed_params), width=100)
    plt.subplots_adjust(bottom=0.25)
    plt.figtext(0.05, 0.03, graphtext, fontsize=8, ha="left")
    # bbox_inches="tight"
    plt.savefig(os.path.join(output_dir, f"{variable_param}-{mode}-course.{IMG_FORMAT}"), format=IMG_FORMAT)


def create_graph_end_sb(course_df, fixed_params, variable_param, variable_param_settings, mode, stats, output_dir, label):
    n_steps = fixed_params["steps"]
    y_label = "proportion utterances non-empty" if mode=="communicated" else "proportion paradigm cells filled"
    df_melted = course_df.melt(id_vars=["timesteps",variable_param], value_vars = stats, value_name = y_label, var_name="statistic")


    # Use last iteration as data
    df_tail = df_melted[df_melted["timesteps"] == n_steps]
    sns.barplot(data=df_tail, x=variable_param, y=y_label, hue="statistic")
    # labels = variable_param_settings  # run_data_means.index  # variable values
    # x = np.arange(len(labels))  # the label locations
    # width = 0.2  # the width of the bars
    # fig, ax = plt.subplots()
    # rects = {}
    # colors = ["deepskyblue", "royalblue", "orange", "darkgoldenrod"]
    # for i, stat in enumerate(stats):
    #     stat_label = stat.replace(
    #         "prop_internal_", "") if mode == "internal" else stat.replace("prop_communicated_", "")
    #     rects[stat] = ax.bar(x+i*width, course_tail_avg[:,stat],
    #                          width=width, edgecolor="white", label=stat_label, color=colors[i])
    # # # Add some text for labels, title and custom x-axis tick labels, etc.
    # if mode == "internal":
    #     ax.set_ylabel('proportion paradigm cells filled')
    # elif mode == "communicated":
    #     ax.set_ylabel('proportion utterances non-empty')
    # ax.set_xlabel(variable_param)
    # ax.set_title(f"{variable_param} ({mode})")
    # ax.set_xticks(x+1.5*width)
    # ax.set_xticklabels(labels)
    # ax.legend()
    # graphtext = textwrap.fill(params_print(fixed_params), width=100)
    # plt.subplots_adjust(bottom=0.25)
    # plt.figtext(0.05, 0.03, graphtext, fontsize=8, ha="left")
    plt.savefig(os.path.join(output_dir, f"{variable_param}-{label}-{mode}-end.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)


def get_course_df_sb(run_data, variable_param, variable_param_settings, stats, mode, output_dir):
    iteration_dfs = []
    for i, row in run_data.iterrows():
        iteration_df = row["datacollector"].get_model_vars_dataframe()[stats]
        iteration_df[variable_param] = row[variable_param]
        iteration_df["run"] = row["Run"]
        # Drop all rows with index 0, since this is a logging artefact
        iteration_df = iteration_df.drop(0)
        iteration_dfs.append(iteration_df)
    course_df = pd.concat(iteration_dfs)
    # Old index (with duplicates because of different param settings and runs) becomes explicit column 'timesteps'
    course_df = course_df.reset_index().rename(columns={"index":"timesteps"})
    course_df.to_csv(os.path.join(output_dir, f"{variable_param}-{mode}-raw.csv"))
    return course_df

def create_graph_course_sb(course_df, fixed_params, variable_param, variable_param_settings, stats, mode, output_dir, label):
    n_steps = fixed_params["steps"]
    y_label = "proportion utterances non-empty" if mode=="communicated" else "proportion paradigm cells filled"
    df_melted = course_df.melt(id_vars=["timesteps",variable_param], value_vars = stats, value_name = y_label, var_name="statistic")
    print(df_melted)
    sns.lineplot(data=df_melted, x="timesteps", y=y_label, hue=variable_param)

    plt.savefig(os.path.join(output_dir, f"{variable_param}-{label}-{mode}-course.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)
    plt.clf()

def rolling_avg(df, window, variable_param, stats):
    # run is unique for combination of run + variable_param, so no need to group also on variable param
    df_rolling = df.copy(deep=True)
    df_rolling[stats] = df.groupby(["run"])[stats].rolling(window=window, min_periods=1).mean().reset_index(level="run",drop=True)
    return df_rolling

def main():
    parser = argparse.ArgumentParser(description='Run agent model from terminal.')
    model_group = parser.add_argument_group('model', 'Model parameters')
    for param in model_params_script:
        model_group.add_argument(f"--{param}", nargs="+",
                                 type=str2bool if param in bool_params else float)
    evaluation_group = parser.add_argument_group('evaluation', 'Evaluation parameters')
    for param in evaluation_params:
        if param in bool_params:
            evaluation_group.add_argument(f'--{param}', action='store_true')
        elif param in string_params:
            evaluation_group.add_argument(f'--{param}', type=str, default=evaluation_params[param])
        else:
            evaluation_group.add_argument(f"--{param}", nargs="+", type=int, default=evaluation_params[param])

    # Parse arguments
    args = vars(parser.parse_args())
    variable_params = {k: v for k, v in args.items() if k in model_params_script and v is not None}
    iterations = args["iterations"]
    steps = args["steps"]


    if (len(iterations) != 1 or len(steps) != 1):
        raise ValueError(
            "Only supply one iterations setting and one steps setting (or none for default).")
    if (len(variable_params) == 0):
        raise ValueError(
            "Pease supply one or more variable parameters.")


    print(f"Evaluating iterations {iterations} and steps {steps}")
    output_dir_custom = OUTPUT_DIR
    if args["runlabel"] != "":
        output_dir_custom = f'{OUTPUT_DIR}-{args["runlabel"]}'
    misc.create_output_dir(output_dir_custom)

    # Try variable parameters one by one, while keeping all of the other parameters fixed
    for var_param, var_param_settings in variable_params.items():
        assert len(iterations) == 1
        assert len(steps) == 1
        iterations_setting = iterations[0]
        steps_setting = steps[0]
        fixed_params = {k: v for k, v in model_params_script.items() if k != var_param}
        fixed_params_print = {**fixed_params, **
                                {"iterations": iterations_setting, "steps": steps_setting}}
        run_data = evaluate_model(fixed_params, {var_param: var_param_settings},
                                    iterations_setting, steps_setting, output_dir=output_dir_custom)
        # create_graph_end(run_data, fixed_params_print, var_param, var_param_settings,
        #                     mode="communicated", stats=stats_communicated, output_dir=output_dir_custom)
        # create_graph_course(run_data, fixed_params_print, var_param, var_param_settings,
        #                     mode="communicated", stats=stats_communicated,
        #                     stat="prop_communicated_suffix_l1", output_dir=output_dir_custom)
        #Seaborn
        course_df = get_course_df_sb(run_data, var_param, var_param_settings, stats_communicated, "communicated", output_dir_custom)
        create_graph_course_sb(course_df, fixed_params_print, var_param, var_param_settings, ["prop_communicated_suffix_l1"], "communicated", output_dir_custom, "raw")
        create_graph_end_sb(course_df, fixed_params_print, var_param, var_param_settings, "communicated", stats_communicated, output_dir_custom, "raw")

        course_df_rolling = rolling_avg(course_df, ROLLING_AVG_WINDOW, var_param, stats_communicated)
        create_graph_course_sb(course_df_rolling, fixed_params_print, var_param, var_param_settings, ["prop_communicated_suffix_l1"], "communicated", output_dir_custom, "rolling")
        create_graph_end_sb(course_df_rolling, fixed_params_print, var_param, var_param_settings, "communicated", stats_communicated, output_dir_custom, "rolling")

    # else:
    #     # Evaluate all combinations of variable parameters
    #     # Only params not changed by user are fixed
    #     fixed_params = {k: v for k, v in model_params_script.items() if k not in variable_params}
    #     for iterations_setting in iterations:
    #         for steps_setting in steps:
    #             run_data = evaluate_model(fixed_params, variable_params,
    #                                       iterations_setting, steps_setting, output_dir_custom)


if __name__ == "__main__":
    main()
