
import evaluation
from agents.config import model_params, STATS_AFTER_STEPS, OUTPUT_DIR
from agents import misc


def test_evaluation_zero_steps():
    # Run model for 0 steps (most probably not evoking stats calculation)
    misc.create_output_dir(OUTPUT_DIR)
    evaluation.evaluate_model(fixed_params=model_params, variable_params={},
                              iterations=1, steps=0, output_dir=OUTPUT_DIR)


def test_evaluation_with_stats():
    # Run as many steps of the simulation to get stats
    misc.create_output_dir(OUTPUT_DIR)
    evaluation.evaluate_model(fixed_params=model_params, variable_params={},
                              iterations=1, steps=STATS_AFTER_STEPS, output_dir=OUTPUT_DIR)
