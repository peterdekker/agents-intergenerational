from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from agents.model import Model
from agents.textbarchart import TextBarChart
from agents import stats

from agents.config import HEIGHT, WIDTH, PROPORTION_L2, SUFFIX_PROB, CAPACITY_L1, CAPACITY_L2, \
    DROP_SUBJECT_PROB, MIN_BOUNDARY_FEATURE_DIST, REDUCTION_HH, NEGATIVE_UPDATE, GENERALIZE_PRODUCTION, \
    GENERALIZE_COMPREHENSION


def draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    colours = stats.compute_colour(agent)
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
    portrayal["Color"] = [colours["suffix"], colours["prefix"]]
    portrayal["stroke_color"] = "rgb(0,0,0)"

    return portrayal


canvas_element = CanvasGrid(draw, HEIGHT, WIDTH, 500, 500)
#dist_chart = ChartModule([{"Label": "global_model_distance", "Color": "Blue"}])
filled_chart = ChartModule([{"Label": "global_filled_prefix_l1", "Color": "Blue"},
                            {"Label": "global_filled_suffix_l1", "Color": "Purple"},
                            {"Label": "global_filled_prefix_l2", "Color": "Orange"},
                            {"Label": "global_filled_suffix_l2", "Color": "Brown"}])
corr_int_chart = ChartModule([{"Label": "avg_proportion_correct_interactions", "Color": "brown"},
                              {"Label": "proportion_correct_interactions", "Color": "green"}])
text_bar_chart = TextBarChart([{"Label": "avg_ambiguity", "Color": "green"},
                               {"Label": "global_affixes_prefix_l1", "Color": "Blue"},
                               {"Label": "global_affixes_suffix_l1", "Color": "Purple"},
                               {"Label": "global_affixes_prefix_l2", "Color": "Orange"},
                               {"Label": "global_affixes_suffix_l2", "Color": "Brown"}])

model_params = {
    "height": HEIGHT,
    "width": WIDTH,
    "proportion_l2": UserSettableParameter("slider", "Proportion L2", PROPORTION_L2, 0.0, 1.0, 0.1),
    "suffix_prob": UserSettableParameter("slider", "Suffix prob (intrans)", SUFFIX_PROB, 0.0, 1.0, 0.1),
    "capacity_l1": UserSettableParameter("slider", "Exemplar capacity L1", CAPACITY_L1, 0, 50, 1),
    "capacity_l2": UserSettableParameter("slider", "Exemplar capacity L2", CAPACITY_L2, 0, 50, 1),
    "drop_subject_prob": UserSettableParameter("slider", "Drop subject prob", DROP_SUBJECT_PROB, 0, 1, 0.1),
    "min_boundary_feature_dist": UserSettableParameter("slider", "Min boundary feature dist",
                                                       MIN_BOUNDARY_FEATURE_DIST, 0, 10, 0.1),
    "reduction_hh": UserSettableParameter('checkbox', 'Reduction H&H', value=REDUCTION_HH),
    "negative_update": UserSettableParameter('checkbox', 'Negative update', value=NEGATIVE_UPDATE),
    "generalize_production": UserSettableParameter("slider", "Generalize production prob",
                                                   GENERALIZE_PRODUCTION, 0, 1, 0.1),
    "generalize_comprehension": UserSettableParameter("slider", "Generalize comprehension prob",
                                                      GENERALIZE_COMPREHENSION, 0, 1, 0.1),
}

server = ModularServer(Model,
                       [canvas_element, filled_chart, corr_int_chart, text_bar_chart],
                       "Contact-induced morphological simplification in Alorese", model_params)
