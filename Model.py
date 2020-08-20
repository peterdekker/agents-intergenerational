from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

import numpy as np

from constants import HEIGHT, WIDTH, MAX_RADIUS, N_AGENTS
import stats

from Agent import Agent


class Model(Model):
    '''
    Model class
    '''

    def __init__(self, height=HEIGHT, width=WIDTH, density=1.0, radius=MAX_RADIUS):
        '''
        Initialize field
        '''

        self.height = height
        self.width = width
        self.density = density
        self.radius = radius

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.global_model_distance = 0.0
        self.correct_interactions = 0.0
        self.steps = 0

        self.datacollector = DataCollector(
            {"global_model_distance": "global_model_distance",
             "proportion_correct_interactions": "proportion_correct_interactions"})

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            if np.random.rand() < self.density:
                agent = Agent((x, y), self)
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
        if self.steps % 1 == 0:
            agents = [a for a, x, y in self.grid.coord_iter()]
            self.global_model_distance = stats.compute_global_dist(agents)

        self.datacollector.collect(self)
