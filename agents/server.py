from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

from agents.model import Model

from agents.config import HEIGHT, WIDTH, model_params_ui


# def draw(agent):
#     '''
#     Portrayal Method for canvas
#     '''
#     if agent is None:
#         return
#     colours = agent.colours
#     portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
#     portrayal["Color"] = [colours["suffix"], colours["prefix"]]
#     portrayal["stroke_color"] = "rgb(0,0,0)"

#     return portrayal


internal_chart = ChartModule([{"Label": "prop_internal_prefix_l1", "Color": "Blue"},
                              {"Label": "prop_internal_suffix_l1", "Color": "Purple"},
                              {"Label": "prop_internal_prefix_l2", "Color": "Orange"},
                              {"Label": "prop_internal_suffix_l2", "Color": "Brown"}])
communicated_chart = ChartModule([{"Label": "prop_communicated_prefix_l1", "Color": "Blue"},
                                  {"Label": "prop_communicated_suffix_l1", "Color": "Yellow"},
                                  {"Label": "prop_communicated_prefix_l2", "Color": "Red"},
                                  {"Label": "prop_communicated_suffix_l2", "Color": "Brown"},
                                  {"Label": "prop_communicated_prefix", "Color": "Purple"},
                                  {"Label": "prop_communicated_suffix", "Color": "Orange"}])
corr_int_chart = ChartModule([{"Label": "avg_proportion_correct_interactions", "Color": "brown"},
                              {"Label": "proportion_correct_interactions", "Color": "green"}])


server = ModularServer(Model,
                       [corr_int_chart, communicated_chart, internal_chart],
                       "Contact-induced morphological simplification iterated", model_params_ui)
