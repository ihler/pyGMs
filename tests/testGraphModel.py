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

  def testConstructor(self):
    gmo = GraphModel()
    self.assertEqual( gmo.nvar, 0)
    self.assertEqual( gmo.nfactors, 0)
    self.assertEqual( gmo.isPairwise(), True)
    self.assertEqual( gmo.isBinary(), True)

    x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
    flist = []
    flist.append( Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
    flist.append( Factor( [x[0],x[4]] , [0.450,0.620,0.80,0.930,0.740,0.180,0.410,0.940,0.920,0.420] ) )
    gmo = GraphModel(flist)
    self.assertEqual( gmo.nvar, 5)
    self.assertEqual( gmo.nfactors, 2)

    self.assertEqual( gmo.isPairwise(), True)
    self.assertEqual( gmo.isBinary(), False)
    #self.assertEqual( gmo.isCSP(), False)


  def testJoint(self):
    x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
    flist = []
    flist.append( Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
    flist.append( Factor( [x[0],x[4]] , [0.450,0.620,0.80,0.930,0.740,0.180,0.410,0.940,0.920,0.420] ) )
    gmo = GraphModel(flist)
    jt = flist[0]*flist[1]
    self.assertTrue( eq_tol( gmo.joint(), jt, tol ) )


  def testHash(self):
    x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
    flist = []
    flist.append( Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
    flist.append( Factor( [x[0],x[4]] , [0.450,0.620,0.80,0.930,0.740,0.180,0.410,0.940,0.920,0.420] ) )
    gmo = GraphModel(flist)
    hash1 = gmo.sig
    gmo.makeMinimal()
    self.assertEqual(hash1, gmo.sig)
    gmo.addFactors( [Factor([x[1],x[3]], [2,1,2,3,2,3] )] )
    self.assertTrue( hash1 != gmo.sig )
    gmo.removeFactors( gmo.factorsWith(x[1]) )
    self.assertEqual(hash1, gmo.sig)




  # TODO: unittests that model copies factors (change factor after & recheck?), unittests copy() function
  def testCopy(self):
    return 


  # unittests value(x) function
  def testValue(self):
    x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
    flist = []
    flist.append( Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
    flist.append( Factor( [x[0],x[4]] , [0.45,0.62,0.80,0.93,0.74,0.18,0.41,0.94,0.92,0.42] ) )
    flist.append( Factor( [x[0],x[1]] , [0.82,0.91,0.13,0.19,0.64,0.10] ) )
    flist.append( Factor( [x[1],x[4]] , [0.46,1.51,0.97,1.17,1.21,0.76,0.81,1.82,1.32,1.48,1.56,1.07,0.80,0.93,0.74] ) )
    gmo = GraphModel(flist)
    jt = flist[0]*flist[1]*flist[2]*flist[3]
    self.assertTrue( abs(        jt[(0,0,0,0)]  - gmo.value(   [0,0,0,0,0]) ) < tol )
    self.assertTrue( abs( np.log(jt[(0,0,0,0)]) - gmo.logValue([0,0,0,0,0]) ) < tol )
    self.assertTrue( abs(        jt[(1,2,0,2)]  - gmo.value(   [1,2,0,0,2]) ) < tol )
    self.assertTrue( abs( np.log(jt[(1,2,0,2)]) - gmo.logValue([1,2,0,0,2]) ) < tol )
    self.assertTrue( abs(        jt[(0,2,1,3)]  - gmo.value(   [0,2,1,0,3]) ) < tol )
    self.assertTrue( abs( np.log(jt[(0,2,1,3)]) - gmo.logValue([0,2,1,0,3]) ) < tol )



  # unittests factorsWith*() functions
  def testFactorsWith(self):
    x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
    flist = []
    flist.append( Factor( [x[0]] , [1.0,1.0] ) )
    flist.append( Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
    flist.append( Factor( [x[0],x[4]] , [0.45,0.62,0.80,0.93,0.74,0.18,0.41,0.94,0.92,0.42] ) )
    flist.append( Factor( [x[0],x[1]] , [0.82,0.91,0.13,0.19,0.64,0.10] ) )
    flist.append( Factor( [x[1],x[4]] , [0.46,1.51,0.97,1.17,1.21,0.76,0.81,1.82,1.32,1.48,1.56,1.07,0.80,0.93,0.74] ) )
    gmo = GraphModel(flist)
 
    self.assertEqual( len(gmo.factorsWith(Var(0,2))) , 4 )    # check # of factors involving X0
    self.assertEqual( len(gmo.factorsWith(0)) , 4 )           # should also work with just ID
    self.assertEqual( len(gmo.factorsWith(Var(2,2))) , 1 )    # same tests for X2
    self.assertEqual( len(gmo.factorsWith(2)) , 1 )           # 
    self.assertEqual( len(gmo.factorsWith(Var(4,5))) , 2 )    # and X4
    self.assertEqual( len(gmo.factorsWith(4)) , 2 )           # 

    self.assertEqual( len(gmo.factorsWithAll([x[0],x[1]])) , 1 )  # check only one factor with x0,x1
    self.assertEqual( len(gmo.factorsWithAll([0,1])) , 1 )
    self.assertEqual( len(gmo.factorsWithAll([x[4],x[1]])) , 1 )  # check only one factor with x4,x1
    self.assertEqual( len(gmo.factorsWithAll([4,1])) , 1 )
    self.assertEqual( len(gmo.factorsWithAll([x[4],x[0],x[1]])) , 0 )  # check no factors with x0,x1,x4
    self.assertEqual( len(gmo.factorsWithAll([4,1,0])) , 0 )

    self.assertEqual( len(gmo.factorsWithAny([x[2],x[1]])) , 3 )  # check factors with x2 or x1
    self.assertEqual( len(gmo.factorsWithAny([2,1])) , 3 )
    self.assertEqual( len(gmo.factorsWithAny([x[4],x[1],x[2]])) , 4 )  # check factors with x2 or x1 or x4
    self.assertEqual( len(gmo.factorsWithAny([4,2,1])) , 4 )

  
  
  # unittests removal of factors: find using "factorsWith" and remove
  def testFactorsWith(self):
    x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
    flist = []
    flist.append( Factor( [x[0]] , [1.0,1.0] ) )
    flist.append( Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
    flist.append( Factor( [x[0],x[4]] , [0.45,0.62,0.80,0.93,0.74,0.18,0.41,0.94,0.92,0.42] ) )
    flist.append( Factor( [x[0],x[1]] , [0.82,0.91,0.13,0.19,0.64,0.10] ) )
    flist.append( Factor( [x[1],x[4]] , [0.46,1.51,0.97,1.17,1.21,0.76,0.81,1.82,1.32,1.48,1.56,1.07,0.80,0.93,0.74] ) )
    gmo = GraphModel(flist)

    gmo.removeFactors( gmo.factorsWithAny([2,1]) )

    self.assertEqual( len(gmo.factorsWithAny([0])) , 2 )
    self.assertEqual( len(gmo.factorsWithAny([4])) , 1 )
    self.assertEqual( len(gmo.factorsWithAny([2,1])) , 0 )

    gmo.addFactors( gmo.factorsWithAny([2,1]) )



  # unittests condition() function : valid config & invalid config
  def testCondition(self):
    x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
    flist = []
    flist.append( Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
    flist.append( Factor( [x[0],x[4]] , [0.45,0.62,0.80,0.93,0.74,0.18,0.41,0.94,0.92,0.42] ) )
    flist.append( Factor( [x[0],x[1]] , [0.82,0.91,0.13,0.19,0.64,0.10] ) )
    flist.append( Factor( [x[1],x[4]] , [0.46,1.51,0.97,1.17,1.21,0.76,0.81,1.82,1.32,1.48,1.56,1.07,0.80,0.93,0.74] ) )
    gmo = GraphModel(flist)
    jt = flist[0]*flist[1]*flist[2]*flist[3]
    gmo.condition({0:1})
    self.assertEqual( len(gmo.factorsWith(1)) , 2 )           # check factor structure of conditional
    self.assertEqual( gmo.factorsWith(1)[0].vars , VarSet([x[1]]) )  
    self.assertTrue( abs(        jt[(1,2,0,2)]  - gmo.value(   [1,2,0,0,2]) ) < tol )  # consistent assign of x0 = no change
    self.assertTrue( abs( np.log(jt[(1,2,0,2)]) - gmo.logValue([1,2,0,0,2]) ) < tol )
    self.assertEqual(gmo.value(   [0,0,0,0,0]), 0.0 )   # incorrect assignment of x0 should produce 0.0

    gmo = GraphModel(flist)
    jt = flist[0]*flist[1]*flist[2]*flist[3]
    gmo.condition({4:1, 2:0})
    self.assertTrue( abs(        jt[(1,2,0,1)]  - gmo.value(   [1,2,0,0,1]) ) < tol )  # consistent assign x2,4 = no change
    self.assertTrue( abs( np.log(jt[(1,2,0,1)]) - gmo.logValue([1,2,0,0,1]) ) < tol )
    self.assertEqual(gmo.value(   [0,0,0,0,0]), 0.0 )   # incorrect assignment of x4 should produce 0.0


  # unittests condition() function : valid config & invalid config
  def testCondition2(self):
    x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
    flist = []
    flist.append( Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
    flist.append( Factor( [x[0],x[4]] , [0.45,0.62,0.80,0.93,0.74,0.18,0.41,0.94,0.92,0.42] ) )
    flist.append( Factor( [x[0],x[1]] , [0.82,0.91,0.13,0.19,0.64,0.10] ) )
    flist.append( Factor( [x[1],x[4]] , [0.46,1.51,0.97,1.17,1.21,0.76,0.81,1.82,1.32,1.48,1.56,1.07,0.80,0.93,0.74] ) )
    gmo = GraphModel(flist)
    jt = flist[0]*flist[1]*flist[2]*flist[3]
    gmo.condition2([x[0]],[1])
    self.assertEqual( len(gmo.factorsWith(1)) , 2 )           # check factor structure of conditional
    self.assertEqual( gmo.factorsWith(1)[0].vars , VarSet([x[1]]) )  
    self.assertTrue( abs(        jt[(1,2,0,2)]  - gmo.value(   [1,2,0,0,2]) ) < tol )  # consistent assign of x0 = no change
    self.assertTrue( abs( np.log(jt[(1,2,0,2)]) - gmo.logValue([1,2,0,0,2]) ) < tol )
    self.assertEqual(gmo.value(   [0,0,0,0,0]), 0.0 )   # incorrect assignment of x0 should produce 0.0

    gmo = GraphModel(flist)
    jt = flist[0]*flist[1]*flist[2]*flist[3]
    gmo.condition2([x[4],x[2]],[1,0])
    self.assertTrue( abs(        jt[(1,2,0,1)]  - gmo.value(   [1,2,0,0,1]) ) < tol )  # consistent assign x2,4 = no change
    self.assertTrue( abs( np.log(jt[(1,2,0,1)]) - gmo.logValue([1,2,0,0,1]) ) < tol )
    self.assertEqual(gmo.value(   [0,0,0,0,0]), 0.0 )   # incorrect assignment of x4 should produce 0.0


  # unittests eliminate() function
  def testEliminate(self):
    x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
    flist = []
    flist.append( Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
    flist.append( Factor( [x[0],x[4]] , [0.45,0.62,0.80,0.93,0.74,0.18,0.41,0.94,0.92,0.42] ) )
    flist.append( Factor( [x[0],x[1]] , [0.82,0.91,0.13,0.19,0.64,0.10] ) )
    flist.append( Factor( [x[1],x[4]] , [0.46,1.51,0.97,1.17,1.21,0.76,0.81,1.82,1.32,1.48,1.56,1.07,0.80,0.93,0.74] ) )
    gmo = GraphModel(flist)
    jt = flist[0]*flist[1]*flist[2]*flist[3]
    elimSum = lambda f,v: f.sum(v)
    gmo.eliminate( [x[0]], elimSum)
    self.assertTrue( abs( jt.sum([x[0]])[(0,0,0)]  - gmo.value( [0,0,0,0,0]) ) < tol )
    self.assertTrue( abs( jt.sum([x[0]])[(1,0,2)]  - gmo.value( [0,1,0,0,2]) ) < tol )
     
    gmo = GraphModel(flist)
    elimMax = lambda f,v: f.max(v)
    gmo.eliminate( [x[0]], elimMax)
    self.assertTrue( abs( jt.max([x[0]])[(0,1,1)]  - gmo.value( [0,0,1,0,1]) ) < tol )
    self.assertTrue( abs( jt.max([x[0]])[(1,1,0)]  - gmo.value( [0,1,1,0,0]) ) < tol )


  def testWildcatter(self):
    dims = [2,2,2,2,3,3,2,3]
    D,D2,D1,D0,R,O,T,Z = (Var(i,dims[i]) for i in range(8))
    
    pT = Factor([T],[1,1])
    pO = Factor([O],[0.5,0.3,0.2])
    pR = Factor([R,O,T],0.0)
    pR[:,0,0] = [1.0/3,1.0/3,1.0/3]
    pR[:,1,0] = [1.0/3,1.0/3,1.0/3]
    pR[:,2,0] = [1.0/3,1.0/3,1.0/3]
    pR[:,0,1] = [0.1,0.3,0.6]
    pR[:,1,1] = [0.3,0.4,0.3]
    pR[:,2,1] = [0.5,0.4,0.1]
    
    pD = Factor([D,D2,D1,D0,R], 0.0)
    pD[0,0,:,:,2] = 1.0
    pD[1,1,:,:,2] = 1.0
    pD[0,:,0,:,1] = 1.0
    pD[1,:,1,:,1] = 1.0
    pD[0,:,:,0,0] = 1.0
    pD[1,:,:,1,0] = 1.0
    
    U0 = Factor([T,Z],1.0)
    U0[:,0] = [70000, 20000]
    
    U1 = Factor([D,Z],1.0)
    U1[:,1] = [70000, 0.0]
    
    U2 = Factor([D,O,Z],1.0)
    U2[0,:,2] = 70000
    U2[1,:,2] = [70000,190000,340000]
    
    factors = [pT,pO,pR,pD,U0,U1,U2]
    
    bn = GraphModel(factors)
    sumElim = lambda F,Xlist: F.sum(Xlist)
    bn.eliminate([O],sumElim)
    bn.eliminate([R],sumElim)
    bn.eliminate([D],sumElim)
    bn.eliminate([Z],sumElim)

    self.assertTrue( bn.joint().argmax() == (1,1,1,0) )
    self.assertTrue( bn.joint().max() == 230000 )

    bn2 = GraphModel(factors)
    bn2.eliminate([O,R,D,Z], sumElim)
    self.assertTrue( bn2.joint().argmax() == (1,1,1,0) )
    self.assertTrue( bn2.joint().max() == 230000 )

    bn3 = GraphModel(factors)
    bn3.condition2([T,D0,D1,D2],[0,1,1,1])
    self.assertTrue( bn3.joint().sum() == 230000 )
     
    bn3 = GraphModel(factors)
    bn3.condition2([T,D0,D1,D2],[0,0,0,0])
    self.assertTrue( bn3.joint().sum() == 210000 )
     


  # unittest reading UAI format (technically "fileformats.py", but uses graphmodel functions)
  def testUaiFormat(self):
    flist = readUai('data/BN_0.uai')
    gmo = GraphModel(flist)
    tup=np.array([[1,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,0,1,1,0,1,0,1,0,1,0,0,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,1,0,1,0,1,1,1,1,0,0,1,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,1,0,1,0,1,],
                  [1,1,1,1,1,0,1,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,0,0,1,1,0,1,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,0,1,],
                  [0,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,1,1,0,],
                  [0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,0,1,1,1,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0,1,0,0,0,0,1,0,1,1,0,0,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1,1,0,0,],
                  [1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,]])
    val=np.array([-84.861237, -94.723917, -83.536594, -90.002882, -93.109856])
    for i in range(tup.shape[0]):
      self.assertTrue( abs(gmo.logValue(tup[i,:]) - val[i] ) < tol )

    evid = readEvidence14('data/BN_0.uai.evid')
    gmo.condition(evid)
    val2=np.array([-np.inf, -np.inf, -83.536594, -90.002882, -93.109856])
    for i in range(tup.shape[0]):
      if val2[i]==-np.inf: self.assertTrue( gmo.value(tup[i,:]) == 0.)
      else: self.assertTrue( abs(gmo.logValue(tup[i,:]) - val2[i] ) < tol )

    return 

  
  # unittest reading WCSP format (technically "fileformats.py", but uses graphmodel functions)
  def testWCSPFormat(self):
    flist,name,ubound = readWCSP('data/fourqueens.wcsp')
    self.assertEqual(name, '4-QUEENS')
    self.assertEqual(ubound, 1.)
    model = GraphModel([-f for f in flist], copy=False, isLog=True)
    self.assertEqual(model.nvar, 4)
    self.assertEqual(model.logValue([1,3,0,2]),  0.)
    self.assertEqual(model.logValue([2,0,3,1]),  0.)
    self.assertEqual(model.logValue([1,3,0,0]), -1.)
    self.assertEqual(model.logValue([1,2,0,2]), -2.)
    self.assertEqual(model.logValue([0,1,2,3]), -6.)
    return 


  # unittest reading Ergo format (technically "fileformats.py", but uses graphmodel functions)
  def testErgoFormat(self):
    flist,names,labels = readErgo('data/iscas89_s386.scan.erg')
    self.assertEqual(len(names), 172)
    self.assertEqual(names[-1], 'G_V9BAR')
    self.assertEqual(len(labels), 172)
    self.assertEqual(labels[-1][0]+';'+labels[1][1], 'X0000;X0001')
    model = GraphModel(flist, copy=False, isLog=False)
    self.assertTrue( model.isBN() )
    self.assertTrue( abs(model.logValue((1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1)) - -9.0109133472792884)<tol )
    self.assertEqual( model.value([0]*172), 0.0 )
    return 



  # unittest variable elimination in BN0 model
  def testUaiBN0_VE(self):
    factors = readUai('data/BN_0.uai')
    evid = readEvidence14('data/BN_0.uai.evid')
    model = GraphModel(factors)
    model.condition(evid)
    ord = [85,41,35,65,47,8,48,58,24,77,55,39,32,30,6,76,80,97,83,59,18,23,38,87,91,81,21,68,29,64,27,9,19,10,11,90,71,88,20,99,0,22,94,37,16,63,17,70,86,13,43,31,1,95,69,96,36,51,4,34,56,15,53,42,28,92,49,74,98,14,45,26,84,40,33,2,73,12,25,60,93,78,46,61,44,89,3,7,54,57,52,50,66,72,62,67,79,82,75,5]
    sumElim = lambda F,Xlist: F.sum(Xlist)
    model.eliminate(ord,sumElim)
    self.assertTrue( eq_tol(model.joint().log(), Factor([],-20.477081), tol) , "lnZ {} != -20.477081".format(model.joint().log()))
    
    model = GraphModel(factors)
    model.condition(evid)
    maxElim = lambda F,Xlist: F.max(Xlist)
    model.eliminate(ord,maxElim)
    self.assertTrue( eq_tol(model.joint().log(), Factor([],-45.105018), tol) , "lnF {} != -45.105018".format(model.joint().log())) 


  # Test junction tree reasoning in BN0 model
  def testUaiBN0_JTree(self):
    factors = readUai('data/BN_0.uai')
    evid = readEvidence14('data/BN_0.uai.evid')
    model = GraphModel(factors)
    model.condition(evid)
    ord = [85,41,35,65,47,8,48,58,24,77,55,39,32,30,6,76,80,97,83,59,18,23,38,87,91,81,21,68,29,64,27,9,19,10,11,90,71,88,20,99,0,22,94,37,16,63,17,70,86,13,43,31,1,95,69,96,36,51,4,34,56,15,53,42,28,92,49,74,98,14,45,26,84,40,33,2,73,12,25,60,93,78,46,61,44,89,3,7,54,57,52,50,66,72,62,67,79,82,75,5]
    from pyGMs.wmb import JTree as JTree
    jt = JTree(model,ord)      # build a junction tree and compute lnZ, p(X13,X53)
    lnZ = jt.msgForward()
    self.assertTrue( abs(lnZ - -20.477081) < tol , "lnZ {} != -20.477081".format(lnZ)) 
    bel = jt.beliefs( [f.vars for f in model.factors] )
    res = Factor([model.X[13],model.X[53]],[[ 0.29651202,  0.02847118], [ 0.15811528,  0.51690152]])
    #print bel[tuple(res.vars)].table
    #print res.table
    self.assertTrue( eq_tol( bel[res.vars], res, tol) )
    lnP,xdraw = jt.sample()    # draw a sample and check its probability under the model
    self.assertTrue( abs(lnP + -20.477081 - model.logValue(xdraw)) < tol ) 
    
    jt = JTree(model,ord,weights=1e-6)
    lnF = jt.msgForward()      # max-product junction tree
    self.assertTrue( abs(lnF - -45.105018) < tol , "lnF {} != -45.105018".format(lnF)) 
    xhat = jt.argmax()         # and find the optimal configuration
    self.assertTrue( abs(model.logValue(xhat) - -45.105018) < tol ) 


if __name__ == '__main__':
  unittest.main()



