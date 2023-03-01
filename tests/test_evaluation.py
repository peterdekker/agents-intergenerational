
import evaluation
from agents.config import model_params_script, RARE_STATS_AFTER_GENERATIONS, OUTPUT_DIR
from agents import misc


def test_evaluation_zero_generations():
    # Run model for 0 generations (most probably not evoking stats calculation)
    misc.create_output_dir(OUTPUT_DIR)
    evaluation.evaluate_model(fixed_params=model_params_script, variable_params={},
                              iterations=1, generations=0, output_dir=OUTPUT_DIR)


def test_evaluation_with_stats():
    # Run as many generations of the simulation to get stats
    misc.create_output_dir(OUTPUT_DIR)
    evaluation.evaluate_model(fixed_params=model_params_script, variable_params={},
                              iterations=1, generations=RARE_STATS_AFTER_GENERATIONS, output_dir=OUTPUT_DIR)
