from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

import numpy as np

from agents.config import N_AGENTS, DATA_FILE, DATA_FILE_SYNTHETIC, MAX_RADIUS, COMMUNICATED_STATS_AFTER_STEPS, RARE_STATS_AFTER_STEPS,\
    CLTS_ARCHIVE_PATH, CLTS_ARCHIVE_URL, CLTS_PATH, COMM_SUCCESS_AFTER_STEPS
from agents import stats
from agents import misc
from agents.agent import Agent
from agents.data import Data


class Model(Model):
    '''
    Model class
    '''

    def __init__(self, n_agents, proportion_l2, 
                 reduction_phonotactics_l1, reduction_phonotactics_l2, alpha_l1, alpha_l2,
                 affix_prior_l1, affix_prior_l2, browser_visualization):
        '''
        Initialize field
        '''
        assert n_agents % 1 == 0
        assert proportion_l2 >= 0 and proportion_l2 <= 1
        assert isinstance(reduction_phonotactics_l1, bool)
        assert isinstance(reduction_phonotactics_l2, bool)
        assert alpha_l1 % 1 == 0
        assert alpha_l2 % 1 == 0
        assert isinstance(affix_prior_l1, bool)
        assert isinstance(affix_prior_l2, bool)
        assert type(browser_visualization) == bool

        self.n_agents = n_agents
        self.proportion_l2 = proportion_l2
        
        self.browser_visualization = browser_visualization


        self.schedule = RandomActivation(self)
        self.steps = 0

        # Agent language model object is created from data file
        self.data = Data(DATA_FILE)
        self.clts = misc.load_clts(CLTS_ARCHIVE_PATH, CLTS_ARCHIVE_URL, CLTS_PATH)

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

        # self.affixes_internal_prefix_l1 = {}
        # self.affixes_internal_suffix_l1 = {}
        # self.affixes_internal_prefix_l2 = {}
        # self.affixes_internal_suffix_l2 = {}

        # Behaviourist
        self.prop_communicated_prefix_l1 = 0.0
        self.prop_communicated_suffix_l1 = 0.0
        self.prop_communicated_prefix_l2 = 0.0
        self.prop_communicated_suffix_l2 = 0.0
        self.prop_communicated_prefix = 0.0
        self.prop_communicated_suffix = 0.0

        self.communicated_prefix_l1 = []
        self.communicated_suffix_l1 = []
        self.communicated_prefix_l2 = []
        self.communicated_suffix_l2 = []
        self.communicated_prefix = []
        self.communicated_suffix = []
        ###

        self.datacollector = DataCollector(
            {  # "internal_model_distance": "internal_model_distance",
                "prop_internal_prefix_l1": "prop_internal_prefix_l1",
                "prop_internal_suffix_l1": "prop_internal_suffix_l1",
                "prop_internal_prefix_l2": "prop_internal_prefix_l2",
                "prop_internal_suffix_l2": "prop_internal_suffix_l2",
                # "affixes_internal_prefix_l1": "affixes_internal_prefix_l1",
                # "affixes_internal_suffix_l1": "affixes_internal_suffix_l1",
                # "affixes_internal_prefix_l2": "affixes_internal_prefix_l2",
                # "affixes_internal_suffix_l2": "affixes_internal_suffix_l2",
                "prop_communicated_prefix_l1": "prop_communicated_prefix_l1",
                "prop_communicated_suffix_l1": "prop_communicated_suffix_l1",
                "prop_communicated_prefix_l2": "prop_communicated_prefix_l2",
                "prop_communicated_suffix_l2": "prop_communicated_suffix_l2",
                "prop_communicated_prefix": "prop_communicated_prefix",
                "prop_communicated_suffix": "prop_communicated_suffix",
                "proportion_correct_interactions": "proportion_correct_interactions",
                "avg_proportion_correct_interactions": "avg_proportion_correct_interactions",
            }
        )

        # Always use same # L2 agents, but randomly divide them
        l2 = misc.spread_l2_agents(self.proportion_l2, N_AGENTS)

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for i in range(self.n_agents):
            agent = Agent(i, self, self.data, init="empty" if l2[i] else "data",
                          affix_prior=affix_prior_l2 if l2[i] else affix_prior_l1,
                          reduction_phonotactics=reduction_phonotactics_l2 if l2[i] else reduction_phonotactics_l1,
                          alpha=alpha_l2 if l2[i] else alpha_l1)
            self.schedule.add(agent)
            self.aggents.append(agent)

        self.agents_l1 = [a for a in self.agents if not a.is_l2()]
        self.agents_l2 = [a for a in self.agents if a.is_l2()]

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        '''
        Run one step of the model.
        '''

        # Reset correct interactions
        self.correct_interactions = 0

        self.schedule.step()

        if self.steps % COMMUNICATED_STATS_AFTER_STEPS == 0:

            # TODO: stats can be calculated upon calling .collect(), by
            # registering methods in datacollector. But then
            # no differentiation between stats calculation intervals is possible

            # Compute proportion non-empty cells in communicative measure
            self.prop_communicated_prefix_l1 = stats.prop_communicated(
                self.communicated_prefix_l1, label="Prefix L1")
            self.prop_communicated_prefix_l2 = stats.prop_communicated(
                self.communicated_prefix_l2, label="Prefix L2")
            self.prop_communicated_suffix_l1 = stats.prop_communicated(
                self.communicated_suffix_l1, label="Suffix L1")
            self.prop_communicated_suffix_l2 = stats.prop_communicated(
                self.communicated_suffix_l2, label="Suffix L2")
            self.prop_communicated_prefix = stats.prop_communicated(
                self.communicated_prefix, label="Prefix")
            self.prop_communicated_suffix = stats.prop_communicated(
                self.communicated_suffix, label="Suffix")

        if self.steps % RARE_STATS_AFTER_STEPS == 0:
            self.prop_internal_prefix_l1 = stats.prop_internal_filled_agents(self.agents_l1, "prefix")
            self.prop_internal_suffix_l1 = stats.prop_internal_filled_agents(self.agents_l1, "suffix")
            self.prop_internal_prefix_l2 = stats.prop_internal_filled_agents(self.agents_l2, "prefix")
            self.prop_internal_suffix_l2 = stats.prop_internal_filled_agents(self.agents_l2, "suffix")

            # self.affixes_internal_suffix_l1 = stats.internal_affix_frequencies_agents(
            #     self.agents_l1, "suffix")
            # self.affixes_internal_prefix_l2 = stats.internal_affix_frequencies_agents(
            #     self.agents_l2, "prefix")
            # self.affixes_internal_suffix_l2 = stats.internal_affix_frequencies_agents(
            #     self.agents_l2, "suffix")
            # self.affixes_internal_prefix_l1 = stats.internal_affix_frequencies_agents(
            #     self.agents_l1, "prefix")
        if self.steps % COMM_SUCCESS_AFTER_STEPS == 0:
            # Now compute proportion of correct interaction
            self.proportion_correct_interactions = self.correct_interactions/float(N_AGENTS)
            self.proportions_correct_interactions.append(self.proportion_correct_interactions)
            self.avg_proportion_correct_interactions = np.mean(self.proportions_correct_interactions)

        self.datacollector.collect(self)
        self.steps += 1
