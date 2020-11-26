from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from Model import Model

from constants import HEIGHT, WIDTH, MAX_RADIUS, PROPORTION_L2, CAPACITY_L1, CAPACITY_L2


def draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    c = agent.colour
    color_str = f"hsl({c[0]},{c[1]}%,{c[2]}%)"
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
    portrayal["Color"] = [color_str, color_str]
    portrayal["stroke_color"] = "rgb(0,0,0)"

    return portrayal


canvas_element = CanvasGrid(draw, HEIGHT, WIDTH, 500, 500)
dist_chart = ChartModule([{"Label": "global_model_distance", "Color": "Blue"}])
corr_int_chart = ChartModule([{"Label": "proportion_correct_interactions", "Color": "Green"}])

model_params = {
    "height": HEIGHT,
    "width": WIDTH,
    "proportion_l2": UserSettableParameter("slider", "Proportion L2", PROPORTION_L2, 0.0, 1.0, 0.1),
    "capacity_l1": UserSettableParameter("slider", "Exemplar capacity L1", CAPACITY_L1, 0, 100, 1),
    "capacity_l2": UserSettableParameter("slider", "Exemplar capacity L2", CAPACITY_L2, 0, 100, 1)
}

server = ModularServer(Model,
                       [canvas_element, dist_chart, corr_int_chart],
                       "Contact-induced morphological simplification in Alorese", model_params)
