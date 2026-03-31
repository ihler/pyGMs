"""
factor.py

Defines variables, variable sets, and dense factors over discrete variables (tables) for graphical models

Version 0.4.1 (2026-03-31)
(c) 2015-2026 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import os
if os.environ.get("PYGMS_USE_TORCH") == "1":
  from pygms.factorTorch import *
else:
  from pygms.factorNumpy import *

