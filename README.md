# IBMQ Manager
A simple tool to schedule batches of quantum circuits to be run in the IBMQ platform, with a simple command line terminal to which an input folder containing OpenQASM files is fed.

This tool allows circuits to be run in job batches of configurable size (batches of 5 for free IBMQ users), and saves the results after each batch is run.

It supports the selection of a varying size of circuit lists per job (usually with a maximum of ~50 circuits per job, depending on the circuit size), different optimization levels, different IBMQ backends and a local Qiskit Aer simulation backend. 

At the moments, in order to save IBMQ result objects containing useful information like device properties and stats, results are saved on `.pkl` files.


## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You'll need Python and the below packages installed:
```
qiskit
tqdm
traceback
```
This can be done by running:

```
pip install qiskit tqdm traceback
```

## Usage instructions
To see the allowed arguments, run:
```
python simulation.py -h
```

From the python file, the arguments' definitions:
```
'indir', type=str, help='Input dir for circuits'
'outdir', type=str, help='Output dir for results'
'backend', type=str, help='Backend to be used'
'--circs_per_job', type=int, default=1, help='provide an integer (default: 5)'
'--optim', type=int, default=0, help='provide an integer (default: 5)'
'--jobs', type=int, default=5, help='provide an integer (default: 5)'
'-t', action="store_true", default=False, help = 'Transpile only'
```

More complete instructions soon...

## Result analysis
A tool was made by the developed that loads the results from the `.pkl` files into a Pandas dataframe, and computes some values of interest, like circuit depth, size, and probability of success.
(To be added)

## Authors

* **Diogo Valada** 
