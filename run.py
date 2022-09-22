# from agents.server import server

# server.launch()

from agents.model import Model
from agents.config import N_AGENTS, PROPORTION_L2, REDUCTION_PHONOTACTICS_L1, REDUCTION_PHONOTACTICS_L2, ALPHA_L1, ALPHA_L2, AFFIX_PRIOR_L1, AFFIX_PRIOR_L2

m = Model(n_agents=N_AGENTS, proportion_l2=PROPORTION_L2,
          reduction_phonotactics_l1=REDUCTION_PHONOTACTICS_L1, reduction_phonotactics_l2=REDUCTION_PHONOTACTICS_L2, alpha_l1=ALPHA_L1, alpha_l2=ALPHA_L2,
          affix_prior_l1=AFFIX_PRIOR_L1, affix_prior_l2=AFFIX_PRIOR_L2, steps=STEPS, n_interactions_per_step=N_INTERACTIONS_PER_STEP, run_id=0)
stats_df = m.run()
print(stats_df)
