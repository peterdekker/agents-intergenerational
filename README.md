# Agents Alorese


## Installation

This script requires Python 3.9 or later. To install the dependencies use pip and the requirements.txt in this directory. e.g.

```
    $ pip install -r requirements.txt
```


## Run evaluation script
The evaluation script can be invoked to run the model from the command line and plot the results to files.

Use the ``--evaluate_prop_l2`` mode to evaluate the effect of the proportion of L2 speakers on both prefix and suffix complexity. Model settings can be set by supplying the setting name and one value, for example the probability of phonotactic reduction and the affix prior probability in this case. In addition, meta-variables generations and iterations can be set. ``--runlabel`` is used to set a custom label for the output file names.

```python3 evaluation.py --evaluate_prop_l2 --generations 100 --iterations 20 --interactions_per_generation 50 --phonotactic_reduction_prob 0.3 --affix_prior_prob 0.2 --runlabel phonred03```

With the ``--evaluate_param`` mode, one can evaluate different values for a specific parameter, and plot the outcomes of the different values as lines in one plot. This is useful to evaluate the affix prior and phonotactic reduction settings. To keep the plot clear, it only shows the suffix L2 complexity. Give the parameter you want to evaluate as option, in this case ``--phonotactic_reduction_prob`` and give a list of settings. No other model parameters can be given at the same time, but meta-parameters (generation, iterations) can be given.
```python3 evaluation.py --evaluate_param --generations 100 --iterations 50 --phonotactic_reduction_prob 0.0 0.25 0.5 0.75 1.0 --runlabel evalphonred```

The ``evaluate_params_heatmap`` mode can be used to evaluate the model for different values of two parameters, for example reduction phonotactics probability and affix prior probability. As output, a heatmap will be plotted where color signifies the slope of suffix L2 complexity over increasing proportions of L2 speakers. Give exactly two model parameters, with a number of parameter settings. In addition, meta-parameters generations and iterations can be given
```python3 evaluation.py --evaluate_params_heatmap --generations 100 --iterations 50 --phonotactic_reduction_prob 0.0 0.25 0.5 0.75 1.0 --affix_prior_prob 0.0 0.25 0.5 0.75 1.0 --runlabel phonredaffixprior```