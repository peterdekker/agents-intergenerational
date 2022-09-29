import argparse
from mesa.batchrunner import BatchRunner, batch_run
# from mesa import batch_run

from agents.model import Model
from agents import misc
from agents.config import model_params_script, evaluation_params, bool_params, string_params, OUTPUT_DIR, IMG_FORMAT, ROLLING_AVG_WINDOW

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from multiprocessing import Pool

# stats_internal = {"prop_internal_prefix_l1": lambda m: m.prop_internal_prefix_l1,
#                   "prop_internal_suffix_l1": lambda m: m.prop_internal_suffix_l1,
#                   "prop_internal_prefix_l2": lambda m: m.prop_internal_prefix_l2,
#                   "prop_internal_suffix_l2": lambda m: m.prop_internal_suffix_l2}


# stats_communicated = {"prop_communicated_prefix_l1": lambda m: m.prop_communicated_prefix_l1,
#                       "prop_communicated_suffix_l1": lambda m: m.prop_communicated_suffix_l1,
#                       "prop_communicated_prefix_l2": lambda m: m.prop_communicated_prefix_l2,
#                       "prop_communicated_suffix_l2": lambda m: m.prop_communicated_suffix_l2}

stats_internal = ["prop_internal_prefix_l1", "prop_internal_suffix_l1",
                  "prop_internal_prefix_l2", "prop_internal_suffix_l2", "prop_internal_prefix", "prop_internal_suffix"]
# stats_communicated = ["prop_communicated_prefix_l1", "prop_communicated_suffix_l1",
#                       "prop_communicated_prefix_l2", "prop_communicated_suffix_l2", "prop_communicated_prefix", "prop_communicated_suffix"]

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


def model_wrapper(arg):
    fixed_params, var_param, var_param_setting, run_id = arg
    all_params = fixed_params | {var_param: var_param_setting}
    m = Model(**all_params, run_id=run_id)
    stats_df = m.run()
    print(f" - {var_param}: {var_param_setting}. Iteration {run_id}.")
    return stats_df


def evaluate_model(fixed_params, var_param, var_param_settings, iterations):
    print(f"Iterations: {iterations}")
    print(f"Variable parameters: {params_print({var_param: var_param_settings})}")
    print(f"Fixed parameters: {params_print(fixed_params)}")

    # runner = BatchRunner(
    #     Model,
    #     variable_params,
    #     fixed_params,
    #     iterations=iterations,
    #     max_steps=steps,
    #     model_reporters={"datacollector": lambda m: m.datacollector}
    # )

    # runner.run_all()

    # run_data_old = runner.get_model_vars_dataframe()
    # print(run_data_old.columns)

    # all_params = fixed_params
    # results = batch_run(Model,
    #                     parameters=all_params,
    #                     number_processes=None,
    #                     iterations=iterations,
    #                     data_collection_period=1,
    #                     max_steps=steps)
    # run_data = pd.DataFrame(results)
    # run_data = run_data.rename(columns={"Step": "timesteps", "RunId": "run"})
    # # Drop first timestep
    # run_data = run_data[run_data.timesteps != 0]

    # dfs = []
    # for var_param_setting in var_param_settings:
    #     print(f" - {var_param}: {var_param_setting}. Iteration: ", end="", flush=True)
    #     all_params = fixed_params | {var_param: var_param_setting}
    #     for i in range(iterations):
    #         m = Model(**all_params, run_id=i)
    #         stats_df = m.run()
    #         dfs.append(stats_df)
    #         print(i, end="|", flush=True)
    #     print("")
    cartesian_var_params_runs = [(fixed_params, var_param, var_param_setting, run_id) for var_param_setting in var_param_settings for run_id in range(iterations)]
    with Pool(processes=None) as pool:
        dfs_multi = pool.map(model_wrapper, cartesian_var_params_runs)
    return pd.concat(dfs_multi).reset_index(drop=True)


def rolling_avg(df, window, stats):
    # run is unique for combination of run + variable_param, so no need to group also on variable param
    df_rolling = df.copy(deep=True)
    df_rolling[stats] = df.groupby(["run"])[stats].rolling(
        window=window, min_periods=1).mean().reset_index(level="run", drop=True)
    return df_rolling


# TODO: Rename stats using ylabel
# TODO: For course, filter stats on only average L1+l2 statistic
def create_graph_course_sb(course_df, variable_param, stats, output_dir, label, runlabel):
    # steps = fixed_params["steps"]
    y_label = "proportion affixes non-empty"
    # y_label = "proportion utterances non-empty" if mode=="communicated" else "proportion paradigm cells filled"
    # df_melted = course_df.melt(id_vars=["timesteps", variable_param],
    #                           value_vars=stats, value_name=y_label, var_name="statistic")
    course_df_stat = course_df[course_df["stat_name"].isin(stats)]
    ax = sns.lineplot(data=course_df_stat, x="timestep", y="stat_value", hue=variable_param)
    ax.set_ylim(0, 1)
    plt.savefig(os.path.join(
        output_dir, f"{variable_param}-{label}-course{'-'+runlabel if runlabel else ''}.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)
    plt.clf()


def create_graph_end_sb(course_df, variable_param, output_dir, label, runlabel):
    y_label = "proportion affixes non-empty"
    # y_label = "proportion utterances non-empty" if mode=="communicated" else "proportion paradigm cells filled"
    # df_melted = course_df.melt(id_vars=["timesteps", variable_param],
    #                           value_vars=stats, value_name=y_label, var_name="statistic")

    # Use last iteration as data
    steps = max(course_df["timestep"])
    df_tail = course_df[course_df["timestep"] == steps]
    ax = sns.lineplot(data=df_tail, x=variable_param, y="stat_value", hue="stat_name")
    ax.set_ylim(0, 1)
    plt.savefig(os.path.join(
        output_dir, f"{variable_param}-{label}-end{'-'+runlabel if runlabel else ''}.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description='Run agent model from terminal.')
    model_group = parser.add_argument_group('model', 'Model parameters')
    for param in model_params_script:
        model_group.add_argument(f"--{param}",
                                 type=str2bool if param in bool_params else float)
    evaluation_group = parser.add_argument_group('evaluation', 'Evaluation parameters')
    for param in evaluation_params:
        if param in bool_params:
            evaluation_group.add_argument(f'--{param}', action='store_true')
        elif param in string_params:
            evaluation_group.add_argument(f'--{param}', type=str, default=evaluation_params[param])
        else:
            evaluation_group.add_argument(f"--{param}", type=float, default=evaluation_params[param])

    # Parse arguments
    args = vars(parser.parse_args())
    # Evaluation params
    iterations = int(args["iterations"])
    runlabel = args["runlabel"]
    plot_from_raw_on = args["plot_from_raw"] != ""

    output_dir_custom = OUTPUT_DIR
    if runlabel != "":
        output_dir_custom = f'{OUTPUT_DIR}-{runlabel}'
    misc.create_output_dir(output_dir_custom)

    # if plot_from_raw_on:
    #     course_df_import = pd.read_csv(plot_from_raw, index_col=0)
    #     # Assume in the imported file, variable parameter was proportion L2
    #     var_param = "proportion_l2"
    #     create_graph_course_sb(course_df_import, var_param, [
    #         "prop_communicated_suffix"], output_dir_custom, "raw", runlabel)
    #     create_graph_end_sb(course_df_import, var_param,
    #                         stats_communicated, output_dir_custom, "raw", runlabel)

    #     course_df_rolling = rolling_avg(course_df_import, ROLLING_AVG_WINDOW, stats_communicated)
    #     create_graph_course_sb(course_df_rolling, var_param, [
    #         "prop_communicated_suffix"], output_dir_custom, "rolling", runlabel)
    #     create_graph_end_sb(course_df_rolling, var_param,
    #                         stats_communicated, output_dir_custom, "rolling", runlabel)

    # If we are running the model, not just plotting from results file
    if not plot_from_raw_on:
        # if (len(iterations) != 1 or len(steps) != 1):
        #     raise ValueError(
        #         "Only supply one iterations setting and one steps setting (or none for default).")
        # Use proportion L2 as variable (independent) param, set given params as fixed params.
        var_param = "proportion_l2"
        var_param_settings = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # assert len(iterations) == 1
        # assert len(steps) == 1

        given_params = {k: v for k, v in args.items() if k in model_params_script and v is not None}
        fixed_params = {
            k: (v if k not in given_params else given_params[k]) for k, v in model_params_script.items() if k != var_param}
        course_df = evaluate_model(fixed_params, var_param, var_param_settings,
                                   iterations)
        course_df.to_csv(os.path.join(output_dir_custom, f"{var_param}-raw.csv"))
        create_graph_course_sb(course_df, var_param, [
            "prop_internal_suffix"], output_dir_custom, "raw", runlabel)
        create_graph_end_sb(course_df, var_param, output_dir_custom, "raw", runlabel)

        # course_df_rolling = rolling_avg(course_df, ROLLING_AVG_WINDOW, stats_internal)
        # create_graph_course_sb(course_df_rolling, var_param, [
        #     "prop_internal_suffix"], output_dir_custom, "rolling", runlabel)
        # create_graph_end_sb(course_df_rolling, var_param,
        #                     stats_internal, output_dir_custom, "rolling", runlabel)


if __name__ == "__main__":
    main()
