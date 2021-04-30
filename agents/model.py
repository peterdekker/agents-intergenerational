from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

import numpy as np
from collections import defaultdict

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
                 negative_update, generalize_production_l1,
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

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.steps = 0

        # Global data object is created from data file
        self.data = Data(DATA_FILE)

        # Stats
        self.proportions_correct_interactions = []
        self.correct_interactions = 0.0
        self.proportion_correct_interactions = 0.0
        self.avg_proportion_correct_interactions = 0.0
        self.ambiguity = defaultdict(list)
        self.avg_ambiguity = {}

        self.global_filled_prefix_l1 = 0.0
        self.global_filled_suffix_l1 = 0.0
        self.global_filled_prefix_l2 = 0.0
        self.global_filled_suffix_l2 = 0.0

        self.global_affixes_prefix_l1 = {}
        self.global_affixes_suffix_l1 = {}
        self.global_affixes_prefix_l2 = {}
        self.global_affixes_suffix_l2 = {}

        self.datacollector = DataCollector(
            {  # "global_model_distance": "global_model_distance",
                "global_filled_prefix_l1": "global_filled_prefix_l1",
                "global_filled_suffix_l1": "global_filled_suffix_l1",
                "global_filled_prefix_l2": "global_filled_prefix_l2",
                "global_filled_suffix_l2": "global_filled_suffix_l2",
                "global_affixes_prefix_l1": "global_affixes_prefix_l1",
                "global_affixes_suffix_l1": "global_affixes_suffix_l1",
                "global_affixes_prefix_l2": "global_affixes_prefix_l2",
                "global_affixes_suffix_l2": "global_affixes_suffix_l2",
                "proportion_correct_interactions": "proportion_correct_interactions",
                "avg_proportion_correct_interactions": "avg_proportion_correct_interactions",
                "avg_ambiguity": "avg_ambiguity"})

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
        self.steps += 1
        '''
        Run one step of the model.
        '''
        # Reset correct interactions
        self.correct_interactions = 0.0

        self.schedule.step()

        # Now compute proportion of correct interaction
        self.proportion_correct_interactions = self.correct_interactions/float(N_AGENTS)
        self.proportions_correct_interactions.append(self.proportion_correct_interactions)
        self.avg_proportion_correct_interactions = np.mean(self.proportions_correct_interactions)
        if self.steps % STATS_AFTER_STEPS == 0:
            # Calculate and reset ambiguity every STAT_AFTER_STEPS_INTERACTIONS,
            # to do some averaging over steps
            self.avg_ambiguity = {k: np.mean(v) for k, v in self.ambiguity.items()}
            self.ambiguity = defaultdict(list)

            agents = [a for a, x, y in self.grid.coord_iter()]
            agents_l1 = [a for a in agents if not a.is_l2()]
            agents_l2 = [a for a in agents if a.is_l2()]
            # TODO: these vars can be calculated upon callig .collect(), by
            # registering methods in datacollector. But then
            # no differentiation between stats calculation intervals is possible
            self.global_filled_prefix_l1 = stats.global_filled(agents_l1, "prefix")
            self.global_filled_suffix_l1 = stats.global_filled(agents_l1, "suffix")
            self.global_filled_prefix_l2 = stats.global_filled(agents_l2, "prefix")
            self.global_filled_suffix_l2 = stats.global_filled(agents_l2, "suffix")

            if self.steps % RARE_STATS_AFTER_STEPS == 0:
                self.global_affixes_prefix_l1 = stats.global_affix_frequencies(agents_l1, "prefix")
                self.global_affixes_suffix_l1 = stats.global_affix_frequencies(agents_l1, "suffix")
                self.global_affixes_prefix_l2 = stats.global_affix_frequencies(agents_l2, "prefix")
                self.global_affixes_suffix_l2 = stats.global_affix_frequencies(agents_l2, "suffix")

        self.datacollector.collect(self)
