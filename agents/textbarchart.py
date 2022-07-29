
# from mesa.visualization.modules import TextElement
# from ascii_graph import Pyasciigraph


# class TextBarChart(TextElement):
#     def __init__(self, fields, data_collector_name="datacollector"):
#         self.fields = fields
#         self.data_collector_name = data_collector_name
#         self.graph = Pyasciigraph(line_length=20,
#                                   min_graph_length=20,
#                                   titlebar="")

#     def render(self, model):
#         data_collector = getattr(model, self.data_collector_name)
#         graphs_rendered = []
#         for s in self.fields:
#             name = s["Label"]
#             try:
#                 val = data_collector.model_vars[name][-1]
#             except(IndexError, KeyError):
#                 raise(ValueError("Could not get model vars."))
#             affixes_items = val.items()
#             graphs_rendered += self.graph.graph(name, affixes_items) + ["<br>"]
#         return "<br>".join(graphs_rendered)
