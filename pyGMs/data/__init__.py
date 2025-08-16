"""
pyGMs.data: Python Graphical Model code, Data submodule

pyGMs submodule for maintaining catalogs of problem instances and downloading them when accessed

Version 0.3.1 (2025-08-15)

(c) 2015-2025 AlexanderIhler under the FreeBSD license; see license.txt for details.
"""

import os, sys
import warnings
from .catalog import Model, Catalog

models = Catalog()

#### import package sources list ###########################################################
try:
  models.add_source_file(os.path.join(os.path.dirname(sys.modules[__name__].__file__),'sources.json'))
except:
  warnings.warn('Unable to process package sources file.')


