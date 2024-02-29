"""
testLimid.py

Unit tests for pyGMs decision models / influence diagrams

Version 0.1.1 (2022-04-06)
(c) 2015- Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import unittest
import numpy as np
import sys
sys.path.append('../../')
from pyGMs import *

def eq_tol(F,G,tolerance):
  if (F.nvar != G.nvar) or (F.vars != G.vars):
    return False
  for x in range(F.numel()):
    if (not (np.isnan(F[x]) and np.isnan(G[x]))) and (np.abs(F[x]-G[x]) > tolerance):
      return False
  return True

tol = 1e-5


def Wildcatter():
  T = Var(3,2)   # Decision "Test" (no/yes)
  O = Var(2,3)   # Random var "OilUnderground" (dry/wet/soak)
  R = Var(1,3)   # Random var "TestResults" (closed/open/nostructure)
  D = Var(0,2)   # Decision "Drill" (no/yes)
  
  pO = Factor([O],[0.5,0.3,0.2])   # p(OilUnderground)
  pR = Factor([R,O,T],0.0)         # p(Results | Oil, Test)
  pR[:,0,0] = [1./3,1./3,1./3]
  pR[:,1,0] = [1./3,1./3,1./3]        # no test = random results
  pR[:,2,0] = [1./3,1./3,1./3]
  pR[:,0,1] = [0.1,0.3,0.6]           # yes test = informative results
  pR[:,1,1] = [0.3,0.4,0.3]           #  closed = lots of oil; open =ok;
  pR[:,2,1] = [0.5,0.4,0.1]           #  nostructure = not much oil
  
  U1 = Factor([T], [0, -10])          # testing costs more than not testing
  U2 = Factor([D,O], 0.)
  U2[0,:] = 0                         # not drilling has no cost and no reward
  U2[1,:] = [-70,50,200]              # drilling reward depends on oil amount
  
  CFactors = [pO,pR]                  # chance factors (CPTs)
  DList = [[T],[R,D]]                 # decisions: T blind, D given R
  UFactors = [U1,U2]                  # utility functions
  return CFactors,DList,UFactors


def Pigs(limid=False):
  H1,H2,H3,H4 = (Var(i,2) for i in [0,1,2,3])  # latent, "healthy at month t" variables
  T1,T2,T3 = (Var(i,2) for i in [4,5,6])       # test results at month t
  D1,D2,D3 = (Var(i,2) for i in [7,8,9])       # decision to treat at month t
  
  pH1 = Factor([H1], [.1,.9])      # pig starts with disease with prob 0.10
  pH12= Factor([H1,H2,D1], 0.)
  pH12[0,:,0] = [.9,.1]               # unhealthy & untreated, stays unhealthy with prob 0.9
  pH12[1,:,0] = [.2,.8]               # healhty & untreated, stays healthy with prob 0.8
  pH12[0,:,1] = [.5,.5]               # unhealthy & treated, cured with prob 0.5
  pH12[1,:,1] = [.1,.9]               # healthy & treated, stays healthy with prob 0.9
  
  pH23 = Factor([H2,H3,D2], pH12.table)  # same probabilities for months 2-3
  pH34 = Factor([H3,H4,D3], pH12.table)  # and months 3-4
  
  pT1 = Factor([H1,T1],0.)         # test health at month t
  pT1[0,:] = [.8,.2]                  # matches true health with prob .8 or .9
  pT1[1,:] = [.1,.9]
  
  pT2 = Factor([H2,T2], pT1.table) # same for months 2 & 3
  pT3 = Factor([H3,T3], pT1.table)
  
  U1 = Factor([D1], [0, -100])     # injecting costs 100 DKK
  U2 = Factor([D2], [0, -100])     # injecting costs 100 DKK
  U3 = Factor([D3], [0, -100])     # injecting costs 100 DKK
  U4 = Factor([H4], [300,1000])    # healthy pigs are worth more in month 4

  CFactors = [pH1,pH12,pH23,pH34,pT1,pT2,pT3]         # chance factors (CPTs)
  if limid==True:
    DList = [[T1,D1],[T1,T2,D1,D2],[T1,T2,T3,D1,D2,D3]] # decisions: D1 depends on T1 & past history
  else: 
    DList = [[T1,D1],[T2,D2],[T3,D3]]                   # decisions: now Dt depends only on Tt
  UFactors = [U1,U2,U3,U4]                            # utility functions
  return CFactors,DList,UFactors



class testLimid(unittest.TestCase):

  def setUp(self):
    return

  def testValuation1(self):
    from valuation import Valuation, factor_to_valuation
    C,D,U = Wildcatter()

    valuations = [factor_to_valuation(f,'P',False) for f in C] + \
             [factor_to_valuation(u,'U',False) for u in U]
    info_arcs = [factor_to_valuation(Factor(d,1.),'P',False) for d in D]
    modelU = GraphModel( valuations + info_arcs )

    modelU.eliminate([2],'sum')
    modelU.eliminate([0],'max')
    modelU.eliminate([1],'sum')
    self.assertTrue( eq_tol( Factor([Var(3,2)],[20,22.5]), modelU.joint().util, 1e-3), "Incorrect solution {} to Wildcatter".format(modelU.joint().util.table) )
    


  def testValuation2(self):
    from valuation import Valuation, factor_to_valuation
    C,D,U = Pigs(False)
    
    valuations = [factor_to_valuation(f,'P',False) for f in C] + \
             [factor_to_valuation(u,'U',False) for u in U]
    info_arcs = [factor_to_valuation(Factor(d,1.),'P',False) for d in D]
    modelU = GraphModel( valuations + info_arcs )
    
    modelU.eliminate([0,1,2,3],'sum')  # cannot observe pig's true health
    modelU.eliminate([9],'max')    # solve for the optimal policy at time 3
    modelU.eliminate([6],'sum')    # average over test observations at time 3
    modelU.eliminate([8],'max')    # select the optimal policy at time 2
    modelU.eliminate([5],'sum')    # and average over test observations at time 2
    modelU.eliminate([7],'max')
    modelU.eliminate([4],'sum')
    self.assertTrue( eq_tol( modelU.joint().util , Factor([],729.225), 1e-3), "Incorrect solution {} to Pigs".format(modelU.joint().util[0]) )


# Read LIMID from file

# Convert LIMID to MMAP & solve
  def testLimid2MMAP(self):
    C,D,U = Wildcatter()
    base = sum([u.min() for u in U])   # ensure positivity
    for u in U: u -= u.min()
    F,Q = Limid2MMAP(C,D,U)
    modelU = GraphModel(F)
    S = modelU.X - VarSet(Q)
    M = VarSet(Q)
    modelU.eliminate(S,'sum')
    modelU.eliminate(M,'max')
    self.assertTrue( eq_tol( modelU.joint()+base, Factor([],22.5), 1e-3 ) )


# Draw LIMID

# TODO: more functions?





if __name__ == '__main__':
  unittest.main()



