import time
import numpy as np
np.random.seed(0)
import sys
sys.path.append('../../')
import pyGMs as gm

N = 12
X = [gm.Var(i,2) for i in range(N**2)];

idx = lambda i,j: i*N+j

fs = []
for i in range(N):
  for j in range(N):
    fs.append( gm.Factor([X[idx(i,j)]],np.random.rand(2)) )
    if i<N-1: fs.append( gm.Factor([X[idx(i,j)],X[idx(i+1,j)]], np.random.rand(2,2)) )
    if j<N-1: fs.append( gm.Factor([X[idx(i,j)],X[idx(i,j+1)]], np.random.rand(2,2)) )


#model = gm.GraphModel(fs)
#sumElim = lambda F,Xlist: F.sum(Xlist)
#model.eliminate( X, sumElim )
#print np.log( model.joint().table )

import pyGMs.wmb
model = gm.GraphModel(fs)
jt = gm.wmb.JTree( model , X)

#print jt
#for b in jt.buckets:
#  for mb in b:
#    print mb," :: ",mb.children
#exit()

t0 = time.time()
print(jt.msgForward())
t1 = time.time()
print("Time ",t1-t0)
t0 = time.time()
#marg = jt.beliefs( [[Xi] for Xi in X] )
marg = jt.beliefs( [f.vars for f in fs] )
t1 = time.time()
print("Time ",t1-t0)


