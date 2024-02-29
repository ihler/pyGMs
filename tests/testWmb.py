"""
testGraphModel.py

Unit tests for pyGMs graphmodel class

Version 0.1.1 (2022-04-06)
(c) 2015 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import unittest
import numpy as np
import sys
sys.path.append('../../')
from pyGMs import *
import pyGMs.wmb as wmb


def eq_tol(F,G,tolerance):
  if (F.nvar != G.nvar) or (F.vars != G.vars):
    return False
  for x in range(F.numel()):
    if (not (np.isnan(F[x]) and np.isnan(G[x]))) and (np.abs(F[x]-G[x]) > tolerance):
      return False
  return True



tol = 1e-5

class testGraphModel(unittest.TestCase):

  def setUp(self):
    return

  def testRandom(self):
    X = [Var(i,3) for i in range(5)]
    fs = [Factor([X[i],X[i+1]],np.random.rand(3,3)) for i in range(4)]
    fs.append( Factor([X[0],X[3]], np.random.rand(3,3)) )
    model = GraphModel(fs)

    inf = wmb.WMB(model, list(range(5)), weights=-1.0 )
    print(inf)
    print(inf.buckets[0])
    print(inf.buckets[1])
    print('Exact: ',np.log(model.joint().sum()))
    print('WMB: ')
    for i in range(25):
        print(inf.msgForward(1.0, 0.25))
        inf.msgBackward()

    bels = inf.msgBackward(beliefs=[f.vars for f in fs])
    for c,f in bels.items(): print(f,f.table)

    def scoreScope(m1,m2):
        print("Scoring ",m1,";",m2)
        if len(m1.clique | m2.clique) > 4: return -1
        mx,mn = max([len(m1.clique),len(m2.clique)]), min([len(m1.clique),len(m2.clique)])
        return float(mx)+float(mn)/mx

    inf.merge(scoreScope)
    print("")
    print(inf)

    inf.addClique(VarSet([X[0],X[1],X[2]]))

    inf.addClique(VarSet([X[0],X[4]]))

    print(inf.buckets[4][0].children)

    self.assertEqual( True, True)



  def testSample(self):
    X = [Var(i,3) for i in range(5)]
    fs = [Factor([X[i],X[i+1]],np.random.rand(3,3)) for i in range(4)]
    fs.append( Factor([X[0],X[3]], np.random.rand(3,3)) )
    model = GraphModel(fs)

    inf = wmb.WMB(model, list(range(5)), weights=1.0 )
    for i in range(25):
        print(inf.msgForward(1.0, 0.25))
        inf.msgBackward()

    bd = inf.msgForward(1.0,0.25)

    w = []
    for s in range(1000):
        q,x = inf.sample()
        w.append( model.logValue(x) - q )

    w = np.array(w)
    mx = w.max()
    w -= mx
    print(mx+np.log( np.exp(w).mean() ))

    self.assertEqual( True, True)

     

if __name__ == '__main__':
  unittest.main()



