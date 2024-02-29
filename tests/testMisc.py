

"""
testMisc.py

Unit tests for additional pyGMs functions

Version 0.1.1 (2022-04-06)
(c) 2015-2021 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import unittest
import numpy as np
import sys
sys.path.append('../../')
import pyGMs as gm

def eq_tol(F,G,tolerance):
        if (F.nvar != G.nvar) or (F.vars != G.vars):
                return False
        for x in range(F.numel()):
                if (not (np.isnan(F[x]) and np.isnan(G[x]))) and (np.abs(F[x]-G[x]) > tolerance):
                        return False
        return True


# TODO:
#



D = np.array([[1,1,1,0,1,1,0,0,1,1,1,1,1,1,0,0,1,0,0,0,0,0,1,1,1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,1,
  1,0,1,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,0,0,1,1,1,0,1,1,0,1,0,0,0,1,1,0,0,1,1,1,0,1,1],
  [1,1,0,0,2,1,0,0,0,2,2,2,0,2,1,1,2,0,2,2,0,2,1,1,2,2,2,0,1,1,1,1,1,0,2,2,2,1,1,1,2,2,0,1,
  1,1,0,0,2,1,1,2,2,0,2,0,2,0,2,0,2,2,0,2,2,0,2,2,1,1,0,2,1,1,2,1,2,0,0,1,0,0,1,2,0,2,2,0,0,1,2,0,2,2,0,1,1,1,2,1],
  [0,1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,1,0,1,1,1,
  1,1,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1],
  [0,2,3,0,0,0,2,3,1,0,3,0,2,3,1,1,3,0,3,3,0,1,0,0,3,1,0,2,1,0,2,3,3,3,3,2,1,2,3,3,3,1,0,0,
  3,2,0,3,3,3,0,0,3,1,2,0,2,3,1,2,1,3,0,3,1,0,3,2,0,3,2,3,2,3,3,2,3,0,0,2,3,0,1,2,3,0,2,0,3,3,0,1,0,1,1,0,2,1,2,2],
  [0,0,1,0,0,0,0,1,1,0,1,1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,1,1,1,0,0,0,1,1,0,0,1,0,1,1,
  0,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,1,0,1]]).T


phat = gm.misc.empirical( [ [X[1]] , X[0:2] , [X[1],X[3],X[4]] ], D )
ptrue= [np.array([29,32,39]), np.array([[13,12,17],[16,20,22]]), 
        np.array([[[5,7],[0,4],[3,2],[3,5]],[[2,7],[2,3],[3,6],[4,5]],[[6,2],[4,4],[5,2],[8,8]]])]
assert(np.max(np.abs(phat[0].table - ptrue[0]))<1e-4)
assert(np.max(np.abs(phat[1].table - ptrue[1]))<1e-4)
assert(np.max(np.abs(phat[2].table - ptrue[2]))<1e-4)

### Fail, data type not supported yet
#D2 = [tuple(d) for d in D]
#phat = gm.misc.empirical( [ [X[1]] , X[0:2] , [X[1],X[3],X[4]] ], D2 )
#assert(np.max(np.abs(phat[0].table - ptrue[0]))<1e-4)
#assert(np.max(np.abs(phat[1].table - ptrue[1]))<1e-4)
#assert(np.max(np.abs(phat[2].table - ptrue[2]))<1e-4)



X = [ gm.Var(0,2), gm.Var(1,3), gm.Var(2,2), gm.Var(3,2), gm.Var(4,5) ]
flist = []
flist.append( gm.Factor( [X[0],X[2]] , [ [0.1,0.9],[0.33,0.67] ] ) )
flist.append( gm.Factor( [X[0],X[4]] , [0.45,0.62,0.80,0.93,0.74,0.18,0.41,0.94,0.92,0.42] ) )
flist.append( gm.Factor( [X[0],X[1]] , [0.82,0.91,0.13,0.19,0.64,0.10] ) )
flist.append( gm.Factor( [X[1],X[4]] , [0.46,1.51,0.97,1.17,1.21,0.76,0.81,1.82,1.32,1.48,1.56,1.07,0.80,0.93,0.74] ) )
flist.append( gm.Factor( [X[3]] , [1,0] ) )
gmo = gm.GraphModel(flist)
jt = flist[0]*flist[1]*flist[2]*flist[3]*flist[4]

D2=[(0, 0, 1,0, 1), (1, 0, 0,0, 4), (1, 0, 0,0, 3), (0, 2, 1,0, 4), (1, 0, 0,0, 3), (0, 1, 1,0, 2), (0, 0, 1,0, 1), (0, 2, 1,0, 3), (0, 0, 1,0, 3), (0, 2, 1,0, 4)]
D = np.array(D2)

jtn = jt/jt.sum();
La = np.array([jtn.log()[tuple(x[v] for v in jtn.vars)] for x in D])
Lb = gm.misc.loglikelihood(gmo,D)
Lc = np.array([-2.58596956, -4.50963573, -3.08882493, -3.15215263, -3.08882493, -4.06386824, -2.58596956, -3.5916054 , -3.01938579, -3.15215263])
assert(np.max(np.abs(La-Lc))<1e-4)
assert(np.max(np.abs(Lb-Lc))<1e-4)


PLb = gm.misc.pseudologlikelihood(gmo,np.array(D))

# TODO: doesn't work for alt data type D2

