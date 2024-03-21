pyGMs : A Python toolbox for Graphical Models
================

This code provides a simple Python-based interface for defining probabilistic
graphical models (Bayesian networks, factor graphs, etc.) over discrete random
variables, along with a number of routines for approximate inference.  It is
being developed for use in teaching, as well as prototyping for research.

The code currently uses [NumPy](http://www.numpy.org/) for representing and
operating on the table-based representation of discrete factors, and
[SortedContainers](https://pypi.python.org/pypi/sortedcontainers) for some
internal representations.  Smaller portions use [networkx](https://networkx.org/)
and [scipy](https://www.scipy.org/) as well.

## Installation

Simply download or clone the repository to a directory *pyGMs*, and add its
parent directory to your Python path, either:
```
$ export PYTHONPATH=${PYTHONPATH}:/directory/containing/
```
or in Python
```
import sys
sys.path.append('/directory/containing/')
```


