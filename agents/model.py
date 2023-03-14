# from mesa import Model
# from mesa.time import RandomActivation
# from mesa.space import SingleGrid
# from mesa.datacollection import DataCollector

from agents.config import N_AGENTS, DATA_FILE, CLTS_ARCHIVE_PATH, CLTS_ARCHIVE_URL, CLTS_PATH, RG
from agents import stats
from agents import misc
from agents.agent import Agent
from agents.data import Data


import numpy as np
import pandas as pd

class Model:
    '''
    Model class
    '''

    def __init__(self, n_agents, proportion_l2,
                 reduction_phonotactics_l1, reduction_phonotactics_l2, alpha_l1, alpha_l2,
                 affix_prior_l1, affix_prior_l2, interaction_l1, interaction_l1_shield_initialization, generations, interactions_per_generation, run_id):
        '''
        Initialize field
        '''
        assert n_agents % 1 == 0
        assert proportion_l2 >= 0 and proportion_l2 <= 1
        assert isinstance(reduction_phonotactics_l1, bool)
        assert isinstance(reduction_phonotactics_l2, bool)
        assert isinstance(affix_prior_l1, bool)
        assert isinstance(affix_prior_l2, bool)
        assert isinstance(interaction_l1, bool)
        assert interaction_l1_shield_initialization % 1 == 0
        assert generations % 1 == 0
        assert interactions_per_generation % 1 == 0

        self.n_agents = int(n_agents)
        self.proportion_l2 = proportion_l2
        self.reduction_phonotactics_l1 = reduction_phonotactics_l1
        self.reduction_phonotactics_l2 = reduction_phonotactics_l2
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.affix_prior_l1 = affix_prior_l1
        self.affix_prior_l2 = affix_prior_l2
        self.interaction_l1 = interaction_l1
        self.interaction_l1_shield_initialization = int(interaction_l1_shield_initialization)
        self.generations = int(generations)
        self.interactions_per_generation = int(interactions_per_generation)
        self.run_id = run_id

        # self.schedule = RandomActivation(self)
        self.current_generation = 0
        self.agents = []

        # Agent language model object is created from data file
        self.data = Data(DATA_FILE, interaction_l1, interaction_l1_shield_initialization)
        self.clts = misc.load_clts(CLTS_ARCHIVE_PATH, CLTS_ARCHIVE_URL, CLTS_PATH)

        self.stats_entries = []

        # Stats
        self.proportions_correct_interactions = []
        self.proportion_correct_interactions = 0.0
        self.avg_proportion_correct_interactions = 0.0

        # self.prop_internal_prefix_l1 = 0.0
        # self.prop_internal_suffix_l1 = 0.0
        # self.prop_internal_prefix_l2 = 0.0
        # self.prop_internal_suffix_l2 = 0.0
        # self.prop_internal_prefix = 0.0
        # self.prop_internal_suffix = 0.0

        # Behaviourist
        # self.prop_communicated_prefix_l1 = 0.0
        # self.prop_communicated_suffix_l1 = 0.0
        # self.prop_communicated_prefix_l2 = 0.0
        # self.prop_communicated_suffix_l2 = 0.0
        # self.prop_communicated_prefix = 0.0
        # self.prop_communicated_suffix = 0.0

        # self.communicated_prefix_l1 = []
        # self.communicated_suffix_l1 = []
        # self.communicated_prefix_l2 = []
        # self.communicated_suffix_l2 = []
        # self.communicated_prefix = []
        # self.communicated_suffix = []
        ###

        # self.datacollector = DataCollector(
        #     {  # "internal_model_distance": "internal_model_distance",
        #         "prop_internal_prefix_l1": "prop_internal_prefix_l1",
        #         "prop_internal_suffix_l1": "prop_internal_suffix_l1",
        #         "prop_internal_prefix_l2": "prop_internal_prefix_l2",
        #         "prop_internal_suffix_l2": "prop_internal_suffix_l2",
        #         # "affixes_internal_prefix_l1": "affixes_internal_prefix_l1",
        #         # "affixes_internal_suffix_l1": "affixes_internal_suffix_l1",
        #         # "affixes_internal_prefix_l2": "affixes_internal_prefix_l2",
        #         # "affixes_internal_suffix_l2": "affixes_internal_suffix_l2",
        #         "prop_communicated_prefix_l1": "prop_communicated_prefix_l1",
        #         "prop_communicated_suffix_l1": "prop_communicated_suffix_l1",
        #         "prop_communicated_prefix_l2": "prop_communicated_prefix_l2",
        #         "prop_communicated_suffix_l2": "prop_communicated_suffix_l2",
        #         "prop_communicated_prefix": "prop_communicated_prefix",
        #         "prop_communicated_suffix": "prop_communicated_suffix",
        #         "proportion_correct_interactions": "proportion_correct_interactions",
        #         "avg_proportion_correct_interactions": "avg_proportion_correct_interactions",
        #     }
        # )
        # self.running = True
        # self.datacollector.collect(self)

    def run(self):

        # Create first generation with only L1 speakers (!), which are instantiated with data
        agents_first_gen = self.create_new_generation(proportion_l2=0.0, init_l1="data", init_l2="empty")
        #  print(self.current_generation, list(map(str, agents_first_gen)))
        self.agents.append(agents_first_gen)

        stats.calculate_internal_stats(agents_first_gen, self.current_generation,
                                       self.proportion_l2, self.stats_entries)
        # agents_l1 = [a for a in agents if not a.is_l2()]
        # agents_l2 = [a for a in agents if a.is_l2()]

        for i in range(self.generations):
            self.generation(self.interactions_per_generation)

        self.stats_df = pd.DataFrame(self.stats_entries)
        self.stats_df["run_id"] = self.run_id

        return self.stats_df

    def create_new_generation(self, proportion_l2, init_l1, init_l2):
        # print("New generation")
        agents = []

        # Always use same # L2 agents, but randomly divide them
        l2_agents = misc.spread_l2_agents(proportion_l2, self.n_agents)
        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents.
        for i in range(self.n_agents):
            agent = Agent(i, self, self.data, init=init_l2 if l2_agents[i] else init_l1,
                          affix_prior=self.affix_prior_l2 if l2_agents[i] else self.affix_prior_l1,
                          reduction_phonotactics=self.reduction_phonotactics_l2 if l2_agents[i] else self.reduction_phonotactics_l1,
                          alpha=self.alpha_l2 if l2_agents[i] else self.alpha_l1,
                          l2=l2_agents[i])
            # self.schedule.add(agent)
            agents.append(agent)
        return agents

    def generation(self, interactions_per_generation):
        '''
        Run one generation of the model: next generation of iterated learning
        '''

        # Reset correct interactions
        self.correct_interactions = 0

        self.current_generation += 1

        # Create next generation of agents, with proportion L2. Both L1 and L2 are empty
        agents_new_gen = self.create_new_generation(
            proportion_l2=self.proportion_l2, init_l1="empty", init_l2="empty")
        agents_new_gen_l1 = [a for a in agents_new_gen if not a.is_l2()]
        agents_new_gen_l2 = [a for a in agents_new_gen if a.is_l2()]

        agents_prev_gen = self.agents[-1]
        agents_prev_gen_l1 = [a for a in agents_prev_gen if not a.is_l2()]
        # agents_prev_gen_l2 = [a for a in agents_prev_gen if a.is_l2()]

        # L1 agents learn directly from a random L1 from the previous generation
        for agent_l1 in agents_new_gen_l1:
            agent_l1.copy_parent(agents_prev_gen_l1)

        # L2 agents learn by being spoken to by previous generation (both L1 and L2)
        for i in range(interactions_per_generation):
            for agent_prev in agents_prev_gen:
                if len(agents_new_gen_l2) > 0:
                    agent_prev.speak(RG.choice(agents_new_gen_l2))

        stats.calculate_internal_stats(agents_new_gen, self.current_generation,
                                       self.proportion_l2, self.stats_entries)

        # Now compute proportion of correct interaction
        self.proportion_correct_interactions = self.correct_interactions/float(N_AGENTS * interactions_per_generation)
        #self.proportions_correct_interactions.append(self.proportion_correct_interactions)
        #self.avg_proportion_correct_interactions = np.mean(self.proportions_correct_interactions)
        stats_entry_prop_correct = {"generation": self.current_generation, "proportion_l2": self.proportion_l2, "stat_name": "prop_correct", "stat_value": self.proportion_correct_interactions}
        self.stats_entries.append(stats_entry_prop_correct)

        # print(self.current_generation, list(map(str, agents_new_gen)))
        self.agents.append(agents_new_gen)

        # L2 agents in generation n choose


        # self.schedule.generation()

        # Compute proportion non-empty cells in communicative measure
        # self.prop_communicated_prefix_l1 = stats.prop_communicated(
        #     self.communicated_prefix_l1, label="Prefix L1")
        # self.prop_communicated_prefix_l2 = stats.prop_communicated(
        #     self.communicated_prefix_l2, label="Prefix L2")
        # self.prop_communicated_suffix_l1 = stats.prop_communicated(
        #     self.communicated_suffix_l1, label="Suffix L1")
        # self.prop_communicated_suffix_l2 = stats.prop_communicated(
        #     self.communicated_suffix_l2, label="Suffix L2")
        # self.prop_communicated_prefix = stats.prop_communicated(
        #     self.communicated_prefix, label="Prefix")
        # self.prop_communicated_suffix = stats.prop_communicated(
        #     self.communicated_suffix, label="Suffix")

        

        # self.datacollector.collect(self)
