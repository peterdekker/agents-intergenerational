from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from Model import Model

from constants import HEIGHT, WIDTH, MAX_RADIUS, PROPORTION_L2


class StatsElement(TextElement):
    '''
    Display a text count of how many happy agents there are.
    '''

    def __init__(self):
        pass

    def render(self, model):
        return f"Global model distance: {model.global_model_distance}"


def draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    c = agent.colour
    color_str = f"rgb({c[0]},{c[1]},{c[2]})"
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
    portrayal["Color"] = [color_str, color_str]
    portrayal["stroke_color"] = "rgb(0,0,0)"

    return portrayal


canvas_element = CanvasGrid(draw, HEIGHT, WIDTH, 500, 500)
dist_element = StatsElement()
dist_chart = ChartModule([{"Label": "global_model_distance", "Color": "Blue"}])
corr_int_chart = ChartModule([{"Label": "proportion_correct_interactions", "Color": "Green"}])

model_params = {
    "height": HEIGHT,
    "width": WIDTH,
    "proportion_l2": UserSettableParameter("slider", "Proportion L2", PROPORTION_L2, 0.0, 1.0, 0.1),
    "radius": UserSettableParameter("slider", "Neighbourhood radius", MAX_RADIUS, 1, MAX_RADIUS, 1),
}

server = ModularServer(Model,
                       [canvas_element, dist_element, dist_chart, corr_int_chart],
                       "Contact-induced morphological simplification in Alorese", model_params)
