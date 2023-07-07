# from mesa import Model
# from mesa.time import RandomActivation
# from mesa.space import SingleGrid
# from mesa.datacollection import DataCollector

from agents.config import N_AGENTS, DATA_FILE, CLTS_ARCHIVE_PATH, CLTS_ARCHIVE_URL, CLTS_PATH, RG
from agents import stats
from agents import misc
from agents.agent import Agent
from agents.data import Data

import pandas as pd


class Model:
    '''
    Model class
    '''

    def __init__(self, n_agents, proportion_l2,
                 reduction_phonotactics_l1, reduction_phonotactics_l2, reduction_phonotactics_prob, reduction_phonotactics_drop_border_phoneme, alpha_l1, alpha_l2,
                 affix_prior_combined_l1, affix_prior_combined_l2, affix_prior_l1, affix_prior_l2, affix_prior_prob, interaction_l1, interaction_l1_shield_initialization, generations, interactions_per_generation, run_id, var_param1_name, var_param1_value, var_param2_name, var_param2_value, output_dir):
        '''
        Initialize field
        '''
        assert n_agents % 1 == 0
        assert proportion_l2 >= 0 and proportion_l2 <= 1
        assert isinstance(reduction_phonotactics_l1, bool)
        assert isinstance(reduction_phonotactics_l2, bool)
        assert isinstance(reduction_phonotactics_drop_border_phoneme, bool)
        assert isinstance(affix_prior_combined_l1, bool)
        assert isinstance(affix_prior_combined_l2, bool)
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
        self.reduction_phonotactics_drop_border_phoneme = reduction_phonotactics_drop_border_phoneme
        self.reduction_phonotactics_prob = reduction_phonotactics_prob
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.affix_prior_combined_l1 = affix_prior_combined_l1
        self.affix_prior_combined_l2 = affix_prior_combined_l2
        self.affix_prior_l1 = affix_prior_l1
        self.affix_prior_l2 = affix_prior_l2
        self.affix_prior_prob = affix_prior_prob
        self.interaction_l1 = interaction_l1
        self.interaction_l1_shield_initialization = int(interaction_l1_shield_initialization)
        self.generations = int(generations)
        self.interactions_per_generation = int(interactions_per_generation)
        self.run_id = run_id
        self.output_dir = output_dir

        # self.schedule = RandomActivation(self)
        self.current_generation = 0

        # Agent language model object is created from data file
        self.data = Data(DATA_FILE, self.interaction_l1, self.interaction_l1_shield_initialization)
        self.clts = misc.load_clts(CLTS_ARCHIVE_PATH, CLTS_ARCHIVE_URL, CLTS_PATH)

        self.stats_entries = []
        self.var_param1_name = var_param1_name
        self.var_param1_value = var_param1_value
        self.var_param2_name = var_param2_name
        self.var_param2_value = var_param2_value

        # Stats

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

        self.correct_interactions = 0
        self.total_interactions = 0

        # Create first generation with only L1 speakers (!), which are instantiated with data
        agents_first_gen = self.create_new_generation(proportion_l2=0.0, init_l1="data", init_l2="empty")
        #  print(self.current_generation, list(map(str, agents_first_gen)))
        self.agents_prev_gen = agents_first_gen

        stats.calculate_internal_stats(agents_first_gen, self.current_generation, self.correct_interactions, self.total_interactions,
                                       self.stats_entries)

        # agents_l1 = [a for a in agents if not a.is_l2()]
        # agents_l2 = [a for a in agents if a.is_l2()]

        for i in range(self.generations):
            self.generation()

        self.stats_df = pd.DataFrame(self.stats_entries)
        self.stats_df["run_id"] = self.run_id
        self.stats_df["proportion_l2"] = self.proportion_l2
        # The parameters to be evaluated, in evaluate_param/evaluate_params_heatmap mode.
        if self.var_param1_name:
            self.stats_df[self.var_param1_name] = self.var_param1_value
        if self.var_param2_name:
            self.stats_df[self.var_param2_name] = self.var_param2_value
        
        # Plot affixes
        if self.proportion_l2 > 0.0:
            stats.affix_sample_diagnosis(self.agents_prev_gen, self.output_dir, self.interactions_per_generation, self.proportion_l2, self.run_id)

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
                          affix_prior_combined=self.affix_prior_combined_l2 if l2_agents[i] else self.affix_prior_combined_l1,
                          affix_prior=self.affix_prior_l2 if l2_agents[i] else self.affix_prior_l1,
                          reduction_phonotactics=self.reduction_phonotactics_l2 if l2_agents[
                              i] else self.reduction_phonotactics_l1,
                          alpha=self.alpha_l2 if l2_agents[i] else self.alpha_l1,
                          l2=l2_agents[i])
            # self.schedule.add(agent)
            agents.append(agent)
        return agents

    def generation(self):
        '''
        Run one generation of the model: next generation of iterated learning
        '''

        # Reset correct interactions
        self.correct_interactions = 0
        self.total_interactions = 0

        self.current_generation += 1

        # Create next generation of agents, with proportion L2. Both L1 and L2 are empty
        agents_new_gen = self.create_new_generation(
            proportion_l2=self.proportion_l2, init_l1="empty", init_l2="empty")
        agents_new_gen_l1 = [a for a in agents_new_gen if not a.is_l2()]

        if self.interaction_l1:
            # When option interaction_l1 is on, all agents interact
            agents_new_gen_interacting = agents_new_gen
        else:
            # In the basic model, only L2 is interacting
            agents_new_gen_interacting = [a for a in agents_new_gen if a.is_l2()]

        agents_prev_gen_l1 = [a for a in self.agents_prev_gen if not a.is_l2()]
        # agents_prev_gen_l2 = [a for a in agents_prev_gen if a.is_l2()]

        # L1 agents learn directly from a random L1 from the previous generation
        for agent_l1 in agents_new_gen_l1:
            agent_l1.copy_parent(agents_prev_gen_l1)

        # L2 agents (or all agents if interaction_l1 on) learn by being spoken to by previous generation (both L1 and L2)
        for i in range(self.interactions_per_generation):
            assert len(self.agents_prev_gen) == N_AGENTS
            for agent_prev in self.agents_prev_gen:
                if len(agents_new_gen_interacting) > 0:
                    agent_prev.speak(RG.choice(agents_new_gen_interacting))

        self.agents_prev_gen = agents_new_gen

        stats.calculate_internal_stats(agents_new_gen, self.current_generation, self.correct_interactions, self.total_interactions,
                                       self.stats_entries)



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
