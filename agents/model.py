from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

import numpy as np

from agents.config import N_AGENTS, DATA_FILE, MAX_RADIUS, STATS_AFTER_STEPS, RARE_STATS_AFTER_STEPS
from agents import stats
from agents import misc
from agents.agent import Agent
from agents.data import Data


class Model(Model):
    '''
    Model class
    '''

    def __init__(self, height, width, proportion_l2, suffix_prob,
                 capacity_l1, capacity_l2, drop_subject_prob,
                 min_boundary_feature_dist, reduction_hh,
                 negative_update, always_affix, balance_prefix_suffix_verbs,
                 fuzzy_match_affix,
                 generalize_production_l1,
                 generalize_production_l2, generalize_update_l1,
                 generalize_update_l2):
        '''
        Initialize field
        '''
        assert height % 1 == 0
        assert width % 1 == 0
        assert proportion_l2 >= 0 and proportion_l2 <= 1
        assert suffix_prob >= 0 and suffix_prob <= 1
        assert capacity_l1 % 1 == 0
        assert capacity_l2 % 1 == 0
        assert drop_subject_prob >= 0 and drop_subject_prob <= 1
        assert min_boundary_feature_dist >= 0
        assert isinstance(reduction_hh, bool)
        assert isinstance(negative_update, bool)
        assert isinstance(always_affix, bool)
        assert isinstance(fuzzy_match_affix, bool)
        assert isinstance(balance_prefix_suffix_verbs, bool)
        assert generalize_production_l1 >= 0 and generalize_production_l1 <= 1
        assert generalize_production_l2 >= 0 and generalize_production_l2 <= 1
        assert generalize_update_l1 >= 0 and generalize_update_l1 <= 1
        assert generalize_update_l2 >= 0 and generalize_update_l2 <= 1

        self.height = height
        self.width = width
        self.proportion_l2 = proportion_l2
        self.radius = MAX_RADIUS
        self.suffix_prob = suffix_prob
        self.drop_subject_prob = drop_subject_prob
        self.min_boundary_feature_dist = min_boundary_feature_dist
        self.reduction_hh = reduction_hh
        self.negative_update = negative_update
        self.always_affix = always_affix
        self.fuzzy_match_affix = fuzzy_match_affix

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.steps = 0

        # Agent language model object is created from data file
        self.data = Data(DATA_FILE, balance_prefix_suffix_verbs)

        # Stats
        self.proportions_correct_interactions = []
        self.proportion_correct_interactions = 0.0
        self.avg_proportion_correct_interactions = 0.0
        # self.ambiguity = defaultdict(list)
        # self.avg_ambiguity = {}

        self.prop_internal_prefix_l1 = 0.0
        self.prop_internal_suffix_l1 = 0.0
        self.prop_internal_prefix_l2 = 0.0
        self.prop_internal_suffix_l2 = 0.0

        self.affixes_internal_prefix_l1 = {}
        self.affixes_internal_suffix_l1 = {}
        self.affixes_internal_prefix_l2 = {}
        self.affixes_internal_suffix_l2 = {}

        # Behaviourist
        self.prop_communicated_prefix_l1 = 0.0
        self.prop_communicated_suffix_l1 = 0.0
        self.prop_communicated_prefix_l2 = 0.0
        self.prop_communicated_suffix_l2 = 0.0

        self.communicated_prefix_l1 = []
        self.communicated_suffix_l1 = []
        self.communicated_prefix_l2 = []
        self.communicated_suffix_l2 = []
        ###

        self.datacollector = DataCollector(
            {  # "internal_model_distance": "internal_model_distance",
                "prop_internal_prefix_l1": "prop_internal_prefix_l1",
                "prop_internal_suffix_l1": "prop_internal_suffix_l1",
                "prop_internal_prefix_l2": "prop_internal_prefix_l2",
                "prop_internal_suffix_l2": "prop_internal_suffix_l2",
                "affixes_internal_prefix_l1": "affixes_internal_prefix_l1",
                "affixes_internal_suffix_l1": "affixes_internal_suffix_l1",
                "affixes_internal_prefix_l2": "affixes_internal_prefix_l2",
                "affixes_internal_suffix_l2": "affixes_internal_suffix_l2",
                "prop_communicated_prefix_l1": "prop_communicated_prefix_l1",
                "prop_communicated_suffix_l1": "prop_communicated_suffix_l1",
                "prop_communicated_prefix_l2": "prop_communicated_prefix_l2",
                "prop_communicated_suffix_l2": "prop_communicated_suffix_l2",
                "proportion_correct_interactions": "proportion_correct_interactions",
                "avg_proportion_correct_interactions": "avg_proportion_correct_interactions",
                # "avg_ambiguity": "avg_ambiguity"
            }
        )

        # Always use same # L2 agents, but randomly divide them
        l2 = misc.spread_l2_agents(self.proportion_l2, N_AGENTS)

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for i, cell in enumerate(self.grid.coord_iter()):
            x = cell[1]
            y = cell[2]
            agent = Agent((x, y), self, self.data, init="empty" if l2[i] else "data",
                          capacity=capacity_l2 if l2[i] else capacity_l1, l2=l2[i],
                          generalize_production=generalize_production_l2 if l2[i] else generalize_production_l1,
                          generalize_update=generalize_update_l2 if l2[i] else generalize_update_l1,)
            self.grid.position_agent(agent, (x, y))
            self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        '''
        Run one step of the model.
        '''

        # Create a new sublist for these stats, for every timestep. So they can be kept separate per timestep
        # TODO: List gets as many items as timesteps, to save memory possibly remove first elements after
        # reaching max len (although this takes O(n) time)
        self.communicated_prefix_l1.append([])
        self.communicated_suffix_l1.append([])
        self.communicated_prefix_l2.append([])
        self.communicated_suffix_l2.append([])

        # Reset correct interactions
        self.correct_interactions = 0

        self.schedule.step()

        if self.steps % STATS_AFTER_STEPS == 0:
            # Now compute proportion of correct interaction
            self.proportion_correct_interactions = self.correct_interactions/float(N_AGENTS)
            self.proportions_correct_interactions.append(self.proportion_correct_interactions)
            self.avg_proportion_correct_interactions = np.mean(self.proportions_correct_interactions)

            # Calculate and reset ambiguity every STAT_AFTER_STEPS_INTERACTIONS,
            # to do some averaging over steps
            # self.avg_ambiguity = {k: np.mean(v) for k, v in self.ambiguity.items()}
            # self.ambiguity = defaultdict(list)

            agents = [a for a, x, y in self.grid.coord_iter()]
            agents_l1 = [a for a in agents if not a.is_l2()]
            agents_l2 = [a for a in agents if a.is_l2()]
            # TODO: stats can be calculated upon calling .collect(), by
            # registering methods in datacollector. But then
            # no differentiation between stats calculation intervals is possible

            # Compute proportion non-empty cells in communicative measure
            self.prop_communicated_prefix_l1 = stats.prop_communicated(
                self.communicated_prefix_l1)
            self.prop_communicated_prefix_l2 = stats.prop_communicated(
                self.communicated_prefix_l2)
            self.prop_communicated_suffix_l1 = stats.prop_communicated(
                self.communicated_suffix_l1)
            self.prop_communicated_suffix_l2 = stats.prop_communicated(
                self.communicated_suffix_l2)

        if self.steps % RARE_STATS_AFTER_STEPS == 0:
            self.prop_internal_prefix_l1 = stats.prop_internal_filled_agents(agents_l1, "prefix")
            self.prop_internal_suffix_l1 = stats.prop_internal_filled_agents(agents_l1, "suffix")
            self.prop_internal_prefix_l2 = stats.prop_internal_filled_agents(agents_l2, "prefix")
            self.prop_internal_suffix_l2 = stats.prop_internal_filled_agents(agents_l2, "suffix")

            self.affixes_internal_prefix_l1 = stats.internal_affix_frequencies_agents(agents_l1, "prefix")
            self.affixes_internal_suffix_l1 = stats.internal_affix_frequencies_agents(agents_l1, "suffix")
            self.affixes_internal_prefix_l2 = stats.internal_affix_frequencies_agents(agents_l2, "prefix")
            self.affixes_internal_suffix_l2 = stats.internal_affix_frequencies_agents(agents_l2, "suffix")
            stats.compute_colours_agents(agents)

        self.datacollector.collect(self)
        self.steps += 1
