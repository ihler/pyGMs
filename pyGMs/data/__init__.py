## Initialization for uai_models wrapper

import os, sys
import warnings
from .catalog import Model, Catalog

models = Catalog()

#### import package sources list ###########################################################
try:
  models.add_source_file(os.path.join(os.path.dirname(sys.modules[__name__].__file__),'sources.json'))
except:
  warnings.warn('Unable to process package sources file.')


