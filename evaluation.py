import argparse
# from mesa.batchrunner import BatchRunner, batch_run
# from mesa import batch_run

from agents.model import Model
from agents import misc
from agents.config import model_params_script, eval_params_script, evaluation_params, bool_params, string_params, OUTPUT_DIR, IMG_FORMAT, ENABLE_MULTITHREADING

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
    fixed_params, var_params, prop_l2_value, iteration, generations = arg
    # Extract variable paramater names, before we add proportion_l2 as another variable parameter
    var_params_items = len(var_params.items())
    vpn1, vpv1 = var_params_items[0]
    if len(var_params_items) == 2:
        vpn2, vpv2 = var_params_items[1]
    else:
        vpn2, vpv2 = None, None
    # add proportion l2 as another variable parameter
    var_params = var_params | {"proportion_l2": prop_l2_value}
    all_params = fixed_params | var_params
    m = Model(**all_params, run_id=iteration, generations=generations, var_param1_name=vpn1,
              var_param1_value=vpv1, var_param2_name=vpn2, var_param2_value=vpv2)
    stats_df = m.run()
    print(f" - {params_print(var_params)}Iteration {iteration}.  Generations: {generations}.")
    return stats_df


def evaluate_model(cartesian_var_params_runs, iterations):
    print(f"Iterations: {iterations}.")

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
def create_graph_course_sb(course_df, variable_param, stat, output_dir, runlabel):
    # generations = fixed_params["generations"]
    y_label = "proportion affixes non-empty"
    course_df_stat = course_df[course_df["stat_name"] == stat]
    course_df_stat = course_df_stat.rename(columns={"stat_value": y_label})
    ax = sns.lineplot(data=course_df_stat, x="generation", y=y_label, hue=variable_param)
    ax.set_ylim(0, 1)
    sns.despine(left=True, bottom=True)
    plt.savefig(os.path.join(
        output_dir, f"{variable_param}-course{'-'+runlabel if runlabel else ''}.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)
    plt.clf()


def create_graph_end_sb(course_df, variable_param, stats, output_dir, runlabel, type):
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
        output_dir, f"{variable_param}-{type}-end{'-'+runlabel if runlabel else ''}.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)
    plt.clf()


def create_heatmap(course_df, variable_param1, variable_param2, stats, output_dir, runlabel):

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
        output_dir, f"{variable_param}-heatmap{'-'+runlabel if runlabel else ''}.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)
    plt.clf()


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
            evaluation_group.add_argument(f'--{param}', type=str, default=eval_params_script[param])
        else:
            evaluation_group.add_argument(f"--{param}", type=int,
                                          default=eval_params_script[param])

    # Parse arguments
    args = vars(parser.parse_args())
    # Evaluation params
    iterations = int(args["iterations"])

    generations = int(args["generations"])
    runlabel = args["runlabel"]
    # Different modes
    plot_from_raw_on = args["plot_from_raw"] != ""
    evaluate_prop_l2 = args["evaluate_prop_l2"]
    evaluate_param = args["evaluate_param"]
    evaluate_params_heatmap = args["evaluate_params_heatmap"]

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
        given_model_params = {k: v for k, v in args.items() if k in model_params_script and v is not None}
        prop_l2_settings = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        if evaluate_prop_l2:
            # Check that only one parameter setting is given per parameter
            assert all([len(v) == 1 for v in given_model_params.values() if v is not None])
            # Use given parameters as fixed parameters, or defaults otherwise. Exclude proportion_l2 to be evaluated.
            fixed_params = {k: (v_default if k not in given_model_params else given_model_params[k][0]) for k, v_default in model_params_script.items(
            ) if k != "proportion_l2"}
        elif evaluate_param or evaluate_params_heatmap:
            if evaluate_param and len(given_model_params) != 1:
                raise ValueError("Exactly 1 parameter has to be given in evaluate_param mode")
            if evaluate_params_heatmap and len(given_model_params) != 2:
                raise ValueError("Exactly 2 parameters have to be given in evaluate_params_heatmap mode")
            # Use all fixed parameters from defaults. given_model_params are variable params to be evaluated, exlucde those and proportion_l2.
            fixed_params = {k: v_default for k, v_default in model_params_script.items(
            ) if k not in given_model_params and k != "proportion_l2"}
        else:
            ValueError("Choose a mode: evaluate_prop_l2 or evaluate_param or evaluate_params_heatmap.")
        print(f"Fixed model parameters: {params_print(fixed_params)}")
        cartesian_var_params_runs = []
        for prop_l2_setting in prop_l2_settings:
            for iteration in range(iterations):
                if evaluate_prop_l2:
                    cp = (fixed_params, {}, prop_l2_setting, iteration, generations)
                    cartesian_var_params_runs.append(cp)
                elif evaluate_param:
                    var_param, var_param_values = list(given_model_params.items())[0]
                    for var_param_value in var_param_values:
                        cp = (fixed_params, {var_param: var_param_value},
                              prop_l2_setting, iteration, generations)
                        cartesian_var_params_runs.append(cp)
                elif evaluate_params_heatmap:
                    var_param1, var_param1_values = list(given_model_params.items())[0]
                    var_param2, var_param2_values = list(given_model_params.items())[1]
                    print(f"Taking into account variable parameters: {var_param1} and {var_param2}")
                    for var_param1_value in var_param1_values:
                        for var_param2_value in var_param2_values:
                            cp = (fixed_params, {
                                  var_param1: var_param1_value, var_param2: var_param2_value}, prop_l2_setting, iteration, generations)
                            cartesian_var_params_runs.append(cp)
        # Old one line loop
        # [({
        #     k: (v if k not in given_params else given_params[k]) for k, v in model_params_script.items() if k != var_param}, var_param, var_param_setting, run_id)
        #     for var_param_setting in var_params[var_param] for var_param in var_params for run_id in range(iterations)]

        course_df = evaluate_model(cartesian_var_params_runs, iterations)
        if evaluate_prop_l2:
            course_df.to_csv(os.path.join(output_dir_custom, "proportion_l2.csv"))
            create_graph_end_sb(course_df, "proportion_l2", stats_internal,
                                output_dir_custom, runlabel, type="complexity")
            create_graph_course_sb(course_df, "proportion_l2",
                                   "prop_internal_suffix", output_dir_custom, "complexity", runlabel)
            # Create extra diagnostic plots for avg #affixes per speaker
            create_graph_end_sb(course_df, "proportion_l2", stats_internal_n_affixes,
                                output_dir_custom, runlabel, type="n_affixes")
            # Create extra diagnostic plots for prop correct interactions
            create_graph_end_sb(course_df, "proportion_l2", stats_prop_correct, output_dir_custom,
                                runlabel, type="prop_correct")
        elif evaluate_param:
            var_param = list(given_model_params.keys())[0]
            course_df.to_csv(os.path.join(output_dir_custom, f"{var_param}-evalparam.csv"))
            create_graph_end_sb(course_df, var_param, ["prop_internal_suffix_l2"],
                                output_dir_custom, runlabel, type="complexity")
        elif evaluate_params_heatmap:
            var_param1 = list(given_model_params.keys())[1]
            var_param2 = list(given_model_params.keys())[2]
            course_df.to_csv(os.path.join(output_dir_custom, f"{var_param1}-{var_param2}-evalparamsheat.csv"))
            create_heatmap(course_df, var_param1, var_param2, ["prop_internal_suffix_l2"],
                           output_dir_custom, runlabel)

        else:
            ValueError("Choose a mode: evaluate_prop_l2 or evaluate_param or evaluate_params_heatmap.")

        # course_df_rolling = rolling_avg(course_df, ROLLING_AVG_WINDOW, stats_internal)
        # create_graph_course_sb(course_df_rolling, var_param, [
        #     "prop_internal_suffix"], output_dir_custom, "rolling", runlabel)


if __name__ == "__main__":
    main()
