"""
factor.py

Defines variables, variable sets, and dense factors over discrete variables (tables) for graphical models

Version 0.3.4 (2025-09-09)
(c) 2015-2025 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import os
if os.environ.get("PYGMS_USE_TORCH") == "1":
  from pyGMs.factorTorch import *
else:
  from pyGMs.factorNumpy import *

