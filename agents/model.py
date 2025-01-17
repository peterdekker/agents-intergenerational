from agents.config import DATA_FILE, CLTS_PATH, RG
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
                 phonotactic_reduction_l1, phonotactic_reduction_l2, phonotactic_reduction_prob, phonotactic_reduction_drop_boundary_phoneme,
                 affix_prior_combined_l1, affix_prior_combined_l2, generalization_l1, generalization_l2, generalization_prob, interaction_l1, interaction_l1_shield_initialization, generations, interactions_per_generation, run_id, var_param1_name, var_param1_value, var_param2_name, var_param2_value, output_dir):
        '''
        Initialize field
        '''
        assert n_agents % 1 == 0
        assert proportion_l2 >= 0 and proportion_l2 <= 1
        assert isinstance(phonotactic_reduction_l1, bool)
        assert isinstance(phonotactic_reduction_l2, bool)
        assert isinstance(phonotactic_reduction_drop_boundary_phoneme, bool)
        assert isinstance(affix_prior_combined_l1, bool)
        assert isinstance(affix_prior_combined_l2, bool)
        assert isinstance(generalization_l1, bool)
        assert isinstance(generalization_l2, bool)
        assert isinstance(interaction_l1, bool)
        assert interaction_l1_shield_initialization % 1 == 0
        assert generations % 1 == 0
        assert interactions_per_generation % 1 == 0

        self.n_agents = int(n_agents)
        self.proportion_l2 = proportion_l2
        self.phonotactic_reduction_l1 = phonotactic_reduction_l1
        self.phonotactic_reduction_l2 = phonotactic_reduction_l2
        self.phonotactic_reduction_drop_boundary_phoneme = phonotactic_reduction_drop_boundary_phoneme
        self.phonotactic_reduction_prob = phonotactic_reduction_prob
        self.affix_prior_combined_l1 = affix_prior_combined_l1
        self.affix_prior_combined_l2 = affix_prior_combined_l2
        self.generalization_l1 = generalization_l1
        self.generalization_l2 = generalization_l2
        self.generalization_prob = generalization_prob
        self.interaction_l1 = interaction_l1
        self.interaction_l1_shield_initialization = int(interaction_l1_shield_initialization)
        self.generations = int(generations)
        self.interactions_per_generation = int(interactions_per_generation)
        self.run_id = run_id
        self.output_dir = output_dir

        self.current_generation = 0

        # Agent language model object is created from data file
        self.data = Data(DATA_FILE, self.interaction_l1, self.interaction_l1_shield_initialization)
        self.clts = misc.load_clts(CLTS_PATH)
        self.cv_pattern_cache = {}

        self.stats_entries = []
        self.var_param1_name = var_param1_name
        self.var_param1_value = var_param1_value
        self.var_param2_name = var_param2_name
        self.var_param2_value = var_param2_value

    def run(self):

        self.correct_interactions = 0
        self.total_interactions = 0

        # Create first generation with only L1 speakers (!), which are instantiated with data
        agents_first_gen = self.create_new_generation(proportion_l2=0.0, init_l1="data", init_l2="empty")
        self.agents_prev_gen = agents_first_gen

        stats.calculate_internal_stats(agents_first_gen, self.current_generation, self.correct_interactions, self.total_interactions,
                                       self.stats_entries)

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

        # Plot affixes, only for the first run
        if self.proportion_l2 > 0.0 and self.run_id == 0:
            stats.affix_sample_diagnosis(self.agents_prev_gen, self.output_dir,
                                         self.interactions_per_generation, self.proportion_l2, self.run_id)

        return self.stats_df

    def create_new_generation(self, proportion_l2, init_l1, init_l2):
        agents = []

        # Always use same # L2 agents, but randomly divide them
        l2_agents = misc.spread_l2_agents(proportion_l2, self.n_agents)
        for i in range(self.n_agents):
            agent = Agent(i, self, self.data, init=init_l2 if l2_agents[i] else init_l1,
                          affix_prior_combined=self.affix_prior_combined_l2 if l2_agents[i] else self.affix_prior_combined_l1,
                          generalization=self.generalization_l2 if l2_agents[i] else self.generalization_l1,
                          phonotactic_reduction=self.phonotactic_reduction_l2 if l2_agents[
                              i] else self.phonotactic_reduction_l1,
                          l2=l2_agents[i])
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

        # L1 agents learn directly from a random L1 from the previous generation
        for agent_l1 in agents_new_gen_l1:
            agent_l1.copy_parent(agents_prev_gen_l1)

        # L2 agents (or all agents if interaction_l1 on) learn by being spoken to by previous generation (both L1 and L2)
        for i in range(self.interactions_per_generation):
            assert len(self.agents_prev_gen) == self.n_agents
            for agent_prev in self.agents_prev_gen:
                if len(agents_new_gen_interacting) > 0:
                    agent_prev.speak(RG.choice(agents_new_gen_interacting))

        self.agents_prev_gen = agents_new_gen

        stats.calculate_internal_stats(agents_new_gen, self.current_generation, self.correct_interactions, self.total_interactions,
                                       self.stats_entries)
