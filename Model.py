from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import math
import numpy as np

from constants import N_AGENTS, DATA_FILE, MAX_RADIUS, RG
import stats
from Agent import Agent
from Data import Data


class Model(Model):
    '''
    Model class
    '''

    def __init__(self, height, width, proportion_l2, suffix_prob,
                 capacity_l1, capacity_l2, drop_subject_prob, drop_object_prob,
                 min_boundary_feature_dist):
        '''
        Initialize field
        '''

        self.height = height
        self.width = width
        self.proportion_l2 = proportion_l2
        self.radius = MAX_RADIUS
        self.capacity_l1 = capacity_l1
        self.capacity_l2 = capacity_l2
        self.suffix_prob = suffix_prob
        self.drop_subject_prob = drop_subject_prob
        self.drop_object_prob = drop_object_prob
        self.min_boundary_feature_dist = min_boundary_feature_dist

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.steps = 0

        # Global data object is created from data file
        self.data = Data(DATA_FILE)

        self.global_model_distance = 0.0
        self.correct_interactions = 0.0
        self.datacollector = DataCollector(
            {"global_model_distance": "global_model_distance",
             "global_filled_prefix_l1": "global_filled_prefix_l1",
             "global_filled_suffix_l1": "global_filled_suffix_l1",
             "global_filled_prefix_l2": "global_filled_prefix_l2",
             "global_filled_suffix_l2": "global_filled_suffix_l2",
             "proportion_correct_interactions": "proportion_correct_interactions"})

        # Always use same # L2 agents, but randomly divide them
        n_l2_agents = math.floor(self.proportion_l2 * N_AGENTS)
        l2 = np.zeros(N_AGENTS)
        l2[0:n_l2_agents] = 1
        RG.shuffle(l2)

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for i, cell in enumerate(self.grid.coord_iter()):
            x = cell[1]
            y = cell[2]
            agent = Agent((x, y), self, self.data, init="empty" if l2[i] else "data",
                          capacity=self.capacity_l2 if l2[i] else self.capacity_l1, l2=l2[i])
            self.grid.position_agent(agent, (x, y))
            self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.steps += 1
        '''
        Run one step of the model.
        '''
        # Set correct interactions to 0 before all interactions are performed
        self.correct_interactions = 0.0
        self.schedule.step()
        # Now compute proportion of correct interaction
        self.proportion_correct_interactions = self.correct_interactions/float(N_AGENTS)
        if self.steps % 10 == 0:
            agents = [a for a, x, y in self.grid.coord_iter()]
            agents_l1 = [a for a in agents if not a.is_l2()]
            agents_l2 = [a for a in agents if a.is_l2()]
            self.global_model_distance = stats.global_dist(agents, self.data.lex_concepts, self.data.persons)
            self.global_filled_prefix_l1 = stats.global_filled(agents_l1, "prefix")
            self.global_filled_suffix_l1 = stats.global_filled(agents_l1, "suffix")
            self.global_filled_prefix_l2 = stats.global_filled(agents_l2, "prefix", )
            self.global_filled_suffix_l2 = stats.global_filled(agents_l2, "suffix")

        self.datacollector.collect(self)
