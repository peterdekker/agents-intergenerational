from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

from agents.model import Model
# from agents.textbarchart import TextBarChart

from agents.config import HEIGHT, WIDTH, model_params_ui


def draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    colours = agent.colours
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
    portrayal["Color"] = [colours["suffix"], colours["prefix"]]
    portrayal["stroke_color"] = "rgb(0,0,0)"

    return portrayal


canvas_element = CanvasGrid(draw, HEIGHT, WIDTH, 500, 500)
#dist_chart = ChartModule([{"Label": "internal_model_distance", "Color": "Blue"}])
internal_chart = ChartModule([{"Label": "prop_internal_prefix_l1", "Color": "Blue"},
                              {"Label": "prop_internal_suffix_l1", "Color": "Purple"},
                              {"Label": "prop_internal_prefix_l2", "Color": "Orange"},
                              {"Label": "prop_internal_suffix_l2", "Color": "Brown"}])
communicated_chart = ChartModule([{"Label": "prop_communicated_prefix_l1", "Color": "Blue"},
                                  {"Label": "prop_communicated_suffix_l1", "Color": "Purple"},
                                  {"Label": "prop_communicated_prefix_l2", "Color": "Orange"},
                                  {"Label": "prop_communicated_suffix_l2", "Color": "Brown"}])
corr_int_chart = ChartModule([{"Label": "avg_proportion_correct_interactions", "Color": "brown"},
                              {"Label": "proportion_correct_interactions", "Color": "green"}])
# text_bar_chart = TextBarChart(
#     {"Label": "affixes_internal_prefix_l1"},
#     {"Label": "affixes_internal_suffix_l1"},
#     {"Label": "affixes_internal_prefix_l2"},
#     {"Label": "affixes_internal_suffix_l2"}])


server = ModularServer(Model,
                       [canvas_element, corr_int_chart, communicated_chart, internal_chart],
                       "Contact-induced morphological simplification in Alorese", model_params_ui)
