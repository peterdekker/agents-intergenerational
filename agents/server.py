from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from agents.model import Model
from agents.textbarchart import TextBarChart

from agents.config import HEIGHT, WIDTH, PROPORTION_L2, SUFFIX_PROB, CAPACITY_L1, CAPACITY_L2, \
    PRONOUN_DROP_PROB, MIN_BOUNDARY_FEATURE_DIST, REDUCTION_HH, NEGATIVE_UPDATE, GENERALIZE_PRODUCTION_L1, \
    GENERALIZE_PRODUCTION_L2, GENERALIZE_UPDATE_L1, GENERALIZE_UPDATE_L2, ALWAYS_AFFIX, BALANCE_PREFIX_SUFFIX_VERBS, \
    UNIQUE_AFFIX, FUZZY_MATCH_AFFIX


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
text_bar_chart = TextBarChart([  # {"Label": "avg_ambiguity", "Color": "green"},
    {"Label": "affixes_internal_prefix_l1"},
    {"Label": "affixes_internal_suffix_l1"},
    {"Label": "affixes_internal_prefix_l2"},
    {"Label": "affixes_internal_suffix_l2"}])

model_params = {
    "height": HEIGHT,
    "width": WIDTH,
    "proportion_l2": UserSettableParameter("slider", "Proportion L2", PROPORTION_L2, 0.0, 1.0, 0.1),
    "suffix_prob": UserSettableParameter("slider", "Suffix prob (intrans)", SUFFIX_PROB, 0.0, 1.0, 0.1),
    "capacity_l1": UserSettableParameter("slider", "Exemplar capacity L1", CAPACITY_L1, 0, 50, 1),
    "capacity_l2": UserSettableParameter("slider", "Exemplar capacity L2", CAPACITY_L2, 0, 50, 1),
    "pronoun_drop_prob": UserSettableParameter("slider", "Pronoun drop prob", PRONOUN_DROP_PROB, 0, 1, 0.1),
    "min_boundary_feature_dist": UserSettableParameter("slider", "Min boundary feature dist",
                                                       MIN_BOUNDARY_FEATURE_DIST, 0, 10, 0.1),
    "reduction_hh": UserSettableParameter('checkbox', 'Reduction H&H', value=REDUCTION_HH),
    "negative_update": UserSettableParameter('checkbox', 'Negative update', value=NEGATIVE_UPDATE),
    "always_affix": UserSettableParameter('checkbox', 'Always affix', value=ALWAYS_AFFIX),
    "balance_prefix_suffix_verbs": UserSettableParameter('checkbox', 'Balance prefix/suffix', value=BALANCE_PREFIX_SUFFIX_VERBS),
    "unique_affix": UserSettableParameter('checkbox', 'Unique affix', value=UNIQUE_AFFIX),
    "fuzzy_match_affix": UserSettableParameter('checkbox', 'Fuzzy match affix', value=FUZZY_MATCH_AFFIX),
    "generalize_production_l1": UserSettableParameter("slider", "Generalize production L1 prob",
                                                      GENERALIZE_PRODUCTION_L1, 0, 1, 0.1),
    "generalize_production_l2": UserSettableParameter("slider", "Generalize production L2 prob",
                                                      GENERALIZE_PRODUCTION_L2, 0, 1, 0.1),
    "generalize_update_l1": UserSettableParameter("slider", "Generalize update L1 prob",
                                                  GENERALIZE_UPDATE_L1, 0, 1, 0.01),
    "generalize_update_l2": UserSettableParameter("slider", "Generalize update L2 prob",
                                                  GENERALIZE_UPDATE_L2, 0, 1, 0.01),
}

server = ModularServer(Model,
                       [canvas_element, corr_int_chart, communicated_chart, internal_chart, text_bar_chart],
                       "Contact-induced morphological simplification in Alorese", model_params)
