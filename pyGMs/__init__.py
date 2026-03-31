"""
pyGMs: Python Graphical Model code

A simple graphical model class for learning about, testing, and developing algorithms
for graphical models.

Version 0.4.0 (2026-03-31)

(c) 2015-2026 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

from sortedcontainers import SortedSet as sset;

from .factor import *
#from .factorSparse import *
from .graphmodel import *
from .filetypes import *
from .misc import *
from .draw import *


__title__ = 'pygms'
__version__ = '0.4.0'
__author__ = 'Alexander Ihler'
__license__ = 'BSD'
__copyright__ = '2015-2026, Alexander Ihler'


