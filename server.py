from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from model import SpeechModel

from constants import HEIGHT, WIDTH, MAX_RADIUS

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
    color = str(max(255-10*agent.interactions, 0))
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
    portrayal["Color"] = [f"rgb({color},0,0)", f"rgb({color},0,0)"]
    portrayal["stroke_color"] = f"rgb({color},0,0)"

    return portrayal


canvas_element = CanvasGrid(draw, HEIGHT, WIDTH, 500, 500)
stats_element = StatsElement()
stats_chart = ChartModule([{"Label": "global_model_distance", "Color": "Black"}])

model_params = {
    "height": HEIGHT,
    "width": WIDTH,
    "density": UserSettableParameter("slider", "Agent density", 1.0, 0.1, 1.0, 0.1),
    "radius": UserSettableParameter("slider", "Neighbourhood radius", MAX_RADIUS, 1, MAX_RADIUS,1),
}

server = ModularServer(SpeechModel,
                       [canvas_element, stats_chart, stats_element],
                       "Agents of speech", model_params)
