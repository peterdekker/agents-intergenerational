
from mesa.visualization.modules import TextElement
from ascii_graph import Pyasciigraph



class TextBarChart(TextElement):
    def __init__(self):
        self.graph = Pyasciigraph(line_length=30, min_graph_length=20)

    def render(self, model):
        affixes_items = model.global_affixes_prefix_l1.items()
        graph_rendered = self.graph.graph("Prefixes L1", affixes_items)
        return "<br>".join(graph_rendered)
        