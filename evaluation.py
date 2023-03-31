import argparse
# from mesa.batchrunner import BatchRunner, batch_run
# from mesa import batch_run

from agents.model import Model
from agents import misc
from agents.config import model_params_script, evaluation_params, bool_params, string_params, OUTPUT_DIR, IMG_FORMAT, ENABLE_MULTITHREADING

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

stats_internal_n_affixes = ["prop_internal_n_affixes_prefix_l1", "prop_internal_n_affixes_suffix_l1",
                            "prop_internal_n_affixes_prefix_l2", "prop_internal_n_affixes_suffix_l2", "prop_internal_n_affixes_prefix", "prop_internal_n_affixes_suffix"]

stats_prop_correct = ["prop_correct"]

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


def evaluate_model(cartesian_var_params_runs, iterations):
    print(f"Iterations: {iterations}")
    print(f"Variable parameters: {params_print(var_params)}")
    
    with Pool(processes=None if ENABLE_MULTITHREADING else 1) as pool:
        dfs_multi = pool.map(model_wrapper, cartesian_var_params_runs)
    return pd.concat(dfs_multi).reset_index(drop=True)


def rolling_avg(df, window, stats):
    # run is unique for combination of run + variable_param, so no need to group also on variable param
    df_rolling = df.copy(deep=True)
    df_rolling[stats] = df.groupby(["run"])[stats].rolling(
        window=window, min_periods=1).mean().reset_index(level="run", drop=True)
    return df_rolling


# TODO: For course, filter stats on only average L1+l2 statistic
def create_graph_course_sb(course_df, variable_param, stat, output_dir, label, runlabel):
    # generations = fixed_params["generations"]
    y_label = "proportion affixes non-empty"
    course_df_stat = course_df[course_df["stat_name"]==stat]
    course_df_stat = course_df_stat.rename(columns={"stat_value": y_label})
    ax = sns.lineplot(data=course_df_stat, x="generation", y=y_label, hue=variable_param)
    ax.set_ylim(0, 1)
    sns.despine(left=True, bottom=True)
    plt.savefig(os.path.join(
        output_dir, f"{variable_param}-{label}-course{'-'+runlabel if runlabel else ''}.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)
    plt.clf()


def create_graph_end_sb(course_df, variable_param, stats, output_dir, label, runlabel, type):
    if type == "complexity":
        y_label = "proportion affixes non-empty"
    elif type == "n_affixes":
        y_label = "n affixes"
    elif type == "prop_correct":
        y_label = "proportion correct interactions"
    else:
        ValueError("Unsupported graph type.")
    # y_label = "proportion utterances non-empty" if mode=="communicated" else "proportion paradigm cells filled"
    # df_melted = course_df.melt(id_vars=["generations", variable_param],
    #                           value_vars=stats, value_name=y_label, var_name="statistic")
    df_stats = course_df[course_df["stat_name"].isin(stats)]
    df_stats = df_stats.rename(columns={"stat_value": y_label})

    # Use last iteration as data
    generations = max(df_stats["generation"])
    df_tail = df_stats[df_stats["generation"] == generations]
    if variable_param == "proportion_l2":
        # evaluate_prop_l2 mode
        # Use different stats as colours
        ax = sns.lineplot(data=df_tail, x="proportion_l2", y=y_label, hue="stat_name")
    else:
        # When evaluate_param mode is on, is variable_param as colour
        ax = sns.lineplot(data=df_tail, x="proportion_l2", y=y_label, hue=variable_param)
    if type == "complexity":  # or type == "prop_correct":
        ax.set_ylim(0, 1)
    sns.despine(left=True, bottom=True)
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
    # Different modes
    plot_from_raw_on = args["plot_from_raw"] != ""
    evaluate_prop_l2 = args["evaluate_prop_l2"]
    evaluate_param = args["evaluate_param"]

    output_dir_custom = OUTPUT_DIR
    if runlabel != "":
        # if evaluate_prop_l2:
        #     runlabel += "-eval_l2"
        # if evaluate_param:
        #     runlabel += "-eval_param"
        output_dir_custom = f'{OUTPUT_DIR}-{runlabel}'
    misc.create_output_dir(output_dir_custom)

    # if plot_from_raw_on:
    # Old code
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
        given_params = {k: v for k, v in args.items() if k in model_params_script and v is not None}
        if evaluate_prop_l2:
            # Use proportion L2 as variable (independent) param, set given params as fixed params.
            var_params = {"proportion_l2": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
        elif evaluate_param:
            if len(given_params)>1
                ValueError("Only 1 parameter can be given in evaluate_param mode")
            var_params = {"proportion_l2": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]} | given_params
        else:
            ValueError("Choose a mode: evaluate_prop_l2 or evaluate_param.")
        cartesian_var_params_runs = []
        for run_id in range(iterations):
            for var_param in var_params:
                # Redetermine fixed parameters for chosen var parameter
                if evaluate_prop_l2:
                    fixed_params = {k: (v_default if k not in given_params else given_params[k]) for k, v_default in model_params_script.items() if k != var_param}
                elif evaluate_param:
                    fixed_params = {k: v for k, v in model_params_script.items() if k != var_param}
                print(f"Var param: {var_param}. Fixed params: {params_print(fixed_params)}")
                for var_param_setting in var_params[var_param]:
                    cp = (fixed_params, var_param, var_param_setting, run_id)
                    cartesian_var_params_runs.append(cp)
        # Old one line loop
        # [({
        #     k: (v if k not in given_params else given_params[k]) for k, v in model_params_script.items() if k != var_param}, var_param, var_param_setting, run_id)
        #     for var_param_setting in var_params[var_param] for var_param in var_params for run_id in range(iterations)]

        course_df = evaluate_model(cartesian_var_params_runs, iterations)
        course_df.to_csv(os.path.join(output_dir_custom, f"{var_param}-raw.csv"))
        if evaluate_prop_l2:
            create_graph_end_sb(course_df, "proportion_l2", stats_internal,
                            output_dir_custom, "raw", runlabel, type="complexity")
            create_graph_course_sb(course_df, "proportion_l2", 
                "prop_internal_suffix", output_dir_custom, "raw", runlabel)
            # Create extra diagnostic plots for avg #affixes per speaker
            create_graph_end_sb(course_df, "proportion_l2", stats_internal_n_affixes,
                                output_dir_custom, "n_affixes_raw", runlabel, type="n_affixes")
            # Create extra diagnostic plots for prop correct interactions
            create_graph_end_sb(course_df, "proportion_l2", stats_prop_correct, output_dir_custom,
                                "prop_correct", runlabel, type="prop_correct")
        elif evaluate_param:
            create_graph_end_sb(course_df, var_param, ["prop_internal_suffix"],
                            output_dir_custom, "raw", runlabel, type="complexity")
        else:
            ValueError("Choose a mode: evaluate_prop_l2 or evaluate_param.")

        # course_df_rolling = rolling_avg(course_df, ROLLING_AVG_WINDOW, stats_internal)
        # create_graph_course_sb(course_df_rolling, var_param, [
        #     "prop_internal_suffix"], output_dir_custom, "rolling", runlabel)


if __name__ == "__main__":
    main()
