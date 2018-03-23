from .factor import *
from .graphmodel import *


# TODO:
# flags in graphical model to ensure no changes to factors ?  how did that work?
# normalize function?  (by max, sum, etc? in place?)
# empirical( data, factor/clique list? ) -- get empirical dists from data
# improve display f'n?
# means & geomeans of functions?
# 

# variable orders -- get statistics (width, memory, pseudotree, height, etc?)
# inference
#   mini-bucket simple versions? dynadecomp versions?
#   message passing: trw primal, dual; mplp, dd, gdd; 
#


def eqtol(A,B,tol=1e-6):
    return (A-B).abs().max() < tol;


def loglikelihood(model, data, logZ=None):
    LL = 0.0;
    if logZ is None: 
        tmp = GraphModel(model.factors)  # copy the graphical model and do VE
        sumElim = lambda F,Xlist: F.sum(Xlist)
        tmp.eliminate( eliminationOrder(model,'wtminfill') , sumElim )
        logZ = tmp.logValue([])
    for s in range(len(data)):
        LL += model.logValue(data[s])
        LL -= logZ
    LL /= len(data)
    return LL


##############################################
def ising_grid(n=10,d=2,sigp=1.0,sigu=0.1):
  '''Return a basic Ising-like grid model.
     n    : size of grid (n x n), default 10
     d    : cardinality of variables (default 2)
     sigp : std dev of log pairwise potentials (non-diagonal terms; default 1.0)
     sigu : std dev of log unary potentials (default 0.1)
  '''
  X = [Var(i,d) for i in range(n**2)]
  E = []
  for i in range(n):
    for j in range(n):
      if (i+1 < n): E.append( (i*n+j,(i+1)*n+j) )
      if (j+1 < n): E.append( (i*n+j,i*n+j+1) )
  fs = [Factor([x],np.exp(sigu*np.random.randn(d))) for x in X]   # unary, then binary factors:
  fs.extend( [Factor([X[i],X[j]],np.exp(sigp*(T+T.T)/np.sqrt(2))) for i,j in E for T in [np.random.randn(d,d)*(1.0-np.eye(d))]] )
  #fs.extend( [Factor([X[i],X[j]],np.exp(sigp*np.random.randn(d,d)*(1.0-np.eye(d)))) for i,j in E] )
  return fs

