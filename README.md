# Agents Alorese


## Installation

This script requires Python 3.9 or later. To install the dependencies use pip and the requirements.txt in this directory. e.g.

```
    $ pip install -r requirements.txt
```


## Run evaluation script
The evaluation script can be invoked to run the model from the command line. It allows running multiple iterations of the model (to average stochasticity), and plot the results with spread in a publishable graph. The script will output a line graph of the morphological prop_nonempty, for different proportions of L2 speakers. Also, the evaluation script runs usually faster than the browser visualization.

The evaluation script always evaluates the proportion L2 speakers as independent variable. As command line arguments, you can give the fixed parameters/dependent variables you want to set. For example, to evaluate the model for different proportions of L2 speakers, while enabling L1 affix prior:

```python3 evaluation.py --affix_prior_combined_l1 True```

This will create an output directory with different graphs and raw result data. An affix can be added to the output directory, for later identification, using the ```--runlabel``` parameter. For example.

```python3 evaluation.py --affix_prior_combined_l1 True --runlabel affpriortest1``` will create an output directory with the current data and time which ends in 'affpriortest1'.

It is also possible to generate graphs later from a raw results file outputted by an earlier run of the script. In that case, use the ```--plot_from_raw``` parameter with the filename of the results file as argument, and without model parameters:

```python3 evaluation.py --plot_from_raw output-2022-03-15-17:12:50.531200/proportion_l2-communicated-raw.csv --runlabel run_from_existing_file```
