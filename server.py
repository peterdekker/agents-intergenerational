from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from Model import Model
import stats

from constants import HEIGHT, WIDTH, MAX_RADIUS, PROPORTION_L2, SUFFIX_PROB, CAPACITY_L1, CAPACITY_L2, DROP_SUBJECT_PROB, DROP_OBJECT_PROB


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
dist_chart = ChartModule([{"Label": "global_model_distance", "Color": "Blue"}])
filled_chart = ChartModule([{"Label": "global_filled_prefix_l1", "Color": "Blue"},
                            {"Label": "global_filled_suffix_l1", "Color": "Purple"},
                            {"Label": "global_filled_prefix_l2", "Color": "Orange"},
                            {"Label": "global_filled_suffix_l2", "Color": "Brown"}])
corr_int_chart = ChartModule([{"Label": "proportion_correct_interactions", "Color": "Green"}])

model_params = {
    "height": HEIGHT,
    "width": WIDTH,
    "proportion_l2": UserSettableParameter("slider", "Proportion L2", PROPORTION_L2, 0.0, 1.0, 0.1),
    "suffix_prob": UserSettableParameter("slider", "Suffix prob (intrans)", SUFFIX_PROB, 0.0, 1.0, 0.1),
    "capacity_l1": UserSettableParameter("slider", "Exemplar capacity L1", CAPACITY_L1, 0, 100, 1),
    "capacity_l2": UserSettableParameter("slider", "Exemplar capacity L2", CAPACITY_L2, 0, 100, 1),
    "drop_subject_prob": UserSettableParameter("slider", "Drop subject prob", DROP_SUBJECT_PROB, 0, 1, 0.1),
    "drop_object_prob": UserSettableParameter("slider", "Drop object prob", DROP_OBJECT_PROB, 0, 1, 0.1)
}

server = ModularServer(Model,
                       [canvas_element, dist_chart, filled_chart, corr_int_chart],
                       "Contact-induced morphological simplification in Alorese", model_params)
