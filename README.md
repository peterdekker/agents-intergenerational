# Agent-based model of phonotactic factors in generational transmission in Alorese

The data used for the model (in folder data) is based on a list of prefixing verbs and prefix and suffix paradigms on page 31-32 of Nishiyama, K., & Kelen, H. (2007). A grammar of Lamaholot, Eastern Indonesia. Lincom Europa.

## Installation

This script requires Python 3.9 or later. To install the dependencies use pip and the requirements.txt in this directory. e.g.

```
    $ pip install -r requirements.txt
```


## Run model
The ``evaluation.py`` script can be invoked to run the model from the command line and plot the results to files. There are two modes to run the script:

Use the ``--evaluate_prop_l2`` mode to evaluate the effect of the proportion of L2 speakers on both prefix and suffix complexity. This mode has been used to generate most plots in the main article. Model settings can be set by supplying the setting name and one value, for example the probability of phonotactic reduction and the generalization probability in this case. In addition, meta-variables generations and iterations can be set. ``--runlabel`` is used to set a custom label for the output file names.

```python3 evaluation.py --evaluate_prop_l2 --generations 100 --iterations 20 --interactions_per_generation 50 --phonotactic_reduction_prob 0.3 --generalization_prob 0.2 --runlabel phonred03```

With the ``--evaluate_param`` mode, one can evaluate different values for a specific parameter, and plot the outcomes of the different values as lines in one plot. This is useful to evaluate different settings for for examle the lexical concept generalization and phonotactic reduction in one plot, as done in the SI of the article. To keep the plot clear, it only shows the suffix L2 complexity. Give the parameter you want to evaluate as option, in this case ``--phonotactic_reduction_prob`` and give a list of settings. No other model parameters can be given at the same time, but meta-parameters (generation, iterations) can be given.
```python3 evaluation.py --evaluate_param --generations 100 --iterations 50 --phonotactic_reduction_prob 0.0 0.25 0.5 0.75 1.0 --runlabel evalphonred```

## Parameters
``config.py`` defines the user-settable parameters that can be given to the evaluation script as command line parameters. Some relevant parameterse are:
 - ``--phonotactic_reduction_prob``: set to 1.0 to turn phonotactic reduction on
 - ``--phonotactic_reduction_drop_border_phoneme``: set to False for full reduction of the affix, set to True for reduction of the border phoneme of the affix
 - ``--generalization_prob``: lexical concept generalisation, set to 1.0 to have full generalisation.