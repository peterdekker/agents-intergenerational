
import evaluation
from agents.config import model_params, STATS_AFTER_STEPS


def test_evaluation_zero_steps():
    # Run model for 0 steps (most probably not evoking stats calculation)
    evaluation.evaluate_model(fixed_params=model_params, variable_params={}, iterations=1, steps=0)


def test_evaluation_with_stats():
    # Run as many steps of the simulation to get stats
    evaluation.evaluate_model(fixed_params=model_params, variable_params={}, iterations=1, steps=STATS_AFTER_STEPS)