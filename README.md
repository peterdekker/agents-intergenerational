# Agents Alorese


## Installation

This script requires Python 3.9 or later. To install the dependencies use pip and the requirements.txt in this directory. e.g.

```
    $ pip install -r requirements.txt
```

## Run in browser

To run the model interactively, to learn about the model, run ``mesa runserver`` in this directory. e.g.

```
    $ mesa runserver
```

Then open your browser to [http://127.0.0.1:8521/](http://127.0.0.1:8521/) and press Reset, then Run.


## Run evaluation script
The evaluation script can be invoked to run the model from the command line. It allows running multiple iterations of the model (to average stochasticity), and plot the results with spread in a publishable graph. The script will output a line graph of the morphological complexity, for different proportions of L2 speakers. Also, the evaluation script runs usually faster than the browser visualization.

The evaluation script always evaluates the proportion L2 speakers as independent variable. As command line arguments, you can give the fixed parameters/dependent variables you want to set. For example, to evaluate the model for different proportions of L2 speakers, while setting the L1 update generalization to 0.5 and the pronoun drop probability to 0.7:

```python3 evaluation.py --gen_update_old_l1 0.5 --pronoun_drop_prob 0.7```

This will create an output directory with different graphs and raw result data. An affix can be added to the output directory, for later identification, using the ```--runlabel``` parameter. For example.

```python3 evaluation.py --gen_update_old_l1 0.5 --runlabel mynewvars``` will create an output directory with the current data and time which ends in 'mynewvars'.

It is also possible to generate graphs later from a raw results file outputted by an earlier run of the script. In that case, use the ```--plot_from_raw``` parameter with the filename of the results file as argument, and without model parameters:

```python3 evaluation.py --plot_from_raw output-2022-03-15-17:12:50.531200/proportion_l2-communicated-raw.csv --runlabel run_from_existing_file```
