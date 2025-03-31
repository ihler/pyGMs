from .factor import *
from .graphmodel import *
import time


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
    """Check for factor equality up to a given numerical tolerance"""
    return (A-B).abs().max() < tol;


def is2D(D):
    """Check if a data set D is a set of configurations (2D) or a single configuration (vector, dict, etc.)
       (checks whether first entry in iterable D is also iterable)
    """
    try:
      first = next(iter(D))
      try:
        next(iter(first))
        return True
      except: 
        return False
    except:
      return False


def timelimit(toc):
  """Iterator that continues to count until the given time limit (in seconds) is reached
  Ex: data = [model.sample() for i in timelimit(10.)]  # generate 10 seconds worth of samples
  """
  x = 0
  tic = time.time()
  while time.time()-tic < toc:
    x += 1
    yield x



################################################################################################
# Simple display functions
#   Numpy arrays and data formats as strings
#

def np2str(nparr,precision=2):
    """Convert a numpy array to a fixed-precision string (useful for debugging)"""
    return np.array_str( nparr, precision=precision)

def np2str1(nparr,precision=2,maxlen=None):
    """Convert a numpy array as a single-line string (useful for debugging)"""
    st = np.array_str( nparr, precision=precision).replace('\n','')
    if (maxlen is not None) and (len(st)>maxlen): st = st[:maxlen] + "..."
    return st

def d2s(cfg,X=None):
  """Convert a data point to a string for printing; missing values as `-`"""
  if is2D(cfg): raise ValueError('cfg should be a single data point, not a 2D set.')
  if isinstance(cfg,tuple): cfg = t2d([cfg])[0]   # convert to dict if necessary
  if X is None: X = range(max(cfg.keys())+1)
  return ''.join(str(cfg[i]) if i in cfg else '-' for i in X);

################################################################################################
# Data conversion functions:
#   Helpful conversions to/from "dict", "tuple", "nparray", 1-hot dataset representations
#

def t2d(T,X=None):
    """Convert a data set from list-of-tuples or nparray to list-of-dict form
    Args:
      T (list[tuple]): Data as [(x0,x1,...) ...]; nan elements indicate missing assignments
      X (list[int]/set[int]): variable indices of the columns of T; default 0..len(T[0])-1
    Return: 
      D (dict[int,int]) : data tuples as dicts, [ { 0:x0, 1:x1, ...} ...], with nans removed
    Ex:  t2d( [(1,nan,0),(0,0,nan)], [0,1,2]) = [{0:1,2:0}, {0:0,1:0}]
    """
    if not is2D(T): raise ValueError(f'Data set is not 2D; use e.g. t2d([x])[0] for single data.')
    if X is None: X = range(len(T[0]))             # default labels: 0...n-1
    D = [{i:xi for i,xi in zip(X,x) if not np.isnan(xi)} for x in T]   # make dict for each tuple
    return D
      
def d2t(D,X=None):
    """Convert a data set from list-of-dict to list-of-tuples form
    Args:
      D (dict[int,int]) : set of assigned variables & values, { i:xi }
      X (list[int]/set[int]): variables to use to index the output tuples; default 0..max(i)-1
    Return: 
      T (list[tuple]): Data D as [(x0,x1,...) ...]; nan elements indicate missing assignments in D

    Ex:  t2d( [{0:3, 1:0},{0:0}] , [0,1,2]) = [(3,0,nan),(0,nan,nan)]
    """
    if not is2D(D): raise ValueError(f'Data set is not 2D; use e.g. d2t([x])[0] for single data.')
    if X is None: X=range( max((int(max(x.keys())) for x in D))+1 ) # max 0...n-1 (max in data)
    T = [tuple(x.get(i,np.nan) for i in X) for x in D]       # make tuples for each dict
    return T 

def onehot(T,X=None):
    """Convert list-of-tuples / nparray data to 1-hot representation (nparray)
    Args:
      T (list[tuple[int]]) : list of m data points [(x0,x1,...) ...]
      X (VarSet) : vars for each column of T; default: estimate Xi's # states from maximum values in T[:,i]
    Return:
      H (nparray shape (m,d)): array of 0/1 values for each element of T, for each possible state of Xi
    """
    if not is2D(T): raise ValueError(f'Data set is not 2D; use e.g. onehot([x])[0] for single data.')
    npT = np.asarray(T);
    try:
      d = [Xi.states for Xi in X]
    except:
      if X is None: X = range(len(T[0]))             # default labels: 0...n-1
      d = np.maximum( (npT.astype(int)+1).max(0), 0 )
    H = np.zeros( (len(T),d.sum()) )
    c = d.cumsum(); c[1:]=c[:-1]; c[0]=0;
    for i in range(len(T)): 
        j = (c+npT[i]); j = j[~np.isnan(j)].astype(int)
        H[i,j] = 1


################################################################################################
# Estimating empirical frequencies
#
def empirical( cliques, data, weights=None, normalize=False):
    """Compute the empirical counts of each configuration of the cliques within the data
       Data should be integer, 0...d; any missing data (`nan') skipped (per clique)
       cliques (list) : list of VarSets; which subsets of variables to examine
       data (dict or (m,n) array) : data[s,i] is the value of x^(s)_i, x_i for sample s
       normalize (bool) : divide by the number of data in each estimate?
    """
    factors = [None]*len(cliques)
    for i,vs in enumerate(cliques):
        vs = VarSet(vs)                     # make sure these are OK & sorted
        try:    vs_data = data[:,[int(v) for v in vs]]      # 2d nparray data
        except: vs_data = np.array([[x[v] for v in vs] for x in data])   # dict or 1d array
        factors[i] = Factor(vs, 0.)
        for s,xs in enumerate(vs_data):
          if np.any(np.isnan(xs)): continue    # skip data with missing entries  
          factors[i].t[ tuple(xs.astype(int)) ] += 1. if weights is None else weights[s]
        if normalize: factors[i].t /= factors[i].sum()
    return factors 


# Empirical "online" estimator object
# Takes (optional) time & number of samples arguments
# Takes generator function returning sample,weight (or just sample? how to tell? weighted=False?) pairs
# Tracks expectations or clique probabilities, along with weight normalizer and weight variance; report neff, etc.
# Allows "restarting" / continuing in an online fashion
# Ex: Empirical( lambda x: x[1]**2/(x[2]+1) ).accumulate(data_iterator, max_time, max_samples)


def __log_empirical( cliques, data, weights=True):
    """Compute the empirical counts of each configuration of the cliques within the data
       Data should be integer, 0...d; any missing data (`nan') skipped (per clique)
       cliques (list) : list of VarSets; which subsets of variables to examine
       data (dict or (m,n) array) : data[s,i] is the value of x^(s)_i, x_i for sample s
       normalize (bool) : divide by the number of data in each estimate?
    """
    factors = { VarSet(c):Factor(VarSet(c),0.) for c in cliques }
    weight_totals = { VarSet(c): 0. for c in cliques }
    for tok in data:
        if weights: x,logw = tok   ## log/exp; also factor init. weights are log or exp?
        else: x,w = tok,1.         ## only data points, weight = 1.
        for a in factors: 
            xa = x[a.labels]
            if np.any(np.isnan(xa)): continue
            factors[a][tuple(xa)] += w   ## TODO: log-sum-exp
            weight_totals[a] += w        ## "" 
        wtot += w




################################################################################################
# Evaluating data likelihood of a model
#

def loglikelihood(model, data, logZ=None):
    """loglikelihood(model, data, logZ): compute the log-likelihood of each data point
       model: GraphModel object (or something with logValue() function)
       data: (m,n) numpy array of m data samples of n variables
       logZ: partition function of model if known (otherwise VE is performed)
    """
    if logZ is None: 
        tmp = GraphModel(model.factors)  # copy the graphical model and do VE
        tmp.eliminate( eliminationOrder(model,'wtminfill')[0] , 'sum' )
        logZ = tmp.logValue([])
    #LL = model.logValue(data) - logZ
    LL = np.array([model.logValue(d) for d in data]) - logZ
    return LL


def pseudologlikelihood(model, data):
    """pseudologlikelihood(model, data): compute the pseudo (log)likelihood value of each data point
       model: GraphModel object (or something with factorsWith() function)
       data: (m,n) numpy array of m data samples of n variables
    """
    # TODO: check isLog?
    def conditional(factor,i,x):   # helper function to compute conditional slice of a factor
        return factor.t[tuple(x[v] if v!=i else slice(v.states) for v in factor.vars)]

    PLL = np.zeros( data.shape )
    for i in range(data.shape[1]):  # for each variable:
        flist = model.factorsWith(i, copy=False)
        for s in range(data.shape[0]):
            pXi = 1.
            for f in flist: pXi *= conditional(f,i,data[s,:])
            PLL[s,i] = np.log( pXi[data[s,i]]/pXi.sum() );
    return PLL.sum(1);


##################################################
# Miscellaneous random graph model constructions:
##################################################
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

# TODO: flag to force ising model to be ferromagnetic? "force_sign = None/+1/-1?"

def boltzmann(theta_ij):
  '''Create a pairwise graphical model from a matrix of parameter values.  
     .. math::
       p(x) \\propto \\exp( \\sum_{i \\neq j} \\theta_{ij} xi xj + \\sum_i \\theta_{ii} xi )
     theta : (n,n) array of parameters
  '''
  n = theta_ij.shape[0]
  X = [Var(i,2) for i in range(n)]
  nzi,nzj = np.nonzero( theta_ij )
  factors = [None]*len(nzi)
  for k,i,j in enumerate(zip(nzi,nzj)):
    if i==j: factors[k] = Factor([X[i]],[0,np.exp(theta[i,i])])
    else:    factors[k] = Factor([X[i],X[j]],[0,0,0,np.exp(theta[i,j])])
  return factors


def random3SAT(alpha,n):
    '''Create a random 3-SAT problem in CNF form
      alpha : float : ratio of variables to clauses
      n     : int   : number of variables
      Returns a list of pyGMs Factors corresponding to each clause.
    '''
    import random
    p = int(round(alpha*n))
    X = [gm.Var(i,2) for i in range(n)] 
    factors = []
    for i in range(p):
        #idx = [np.random.randint(n), np.random.randint(n), np.random.randint(n)]
        idx = random.sample(range(n),3)
        f = gm.Factor([X[idx[0]],X[idx[1]],X[idx[2]]],1.0)
        cfg = [np.random.randint(2), np.random.randint(2), np.random.randint(2)]
        f[cfg[0],cfg[1],cfg[2]] = 0.0
        factors.append(f)
    return factors

## TODO: add random spatial graph generators? (Table distributions?)


def fit_chowliu(data, penalty=0, weights=None):
    """Select a maximum likelihood tree-structured graph & parameters
      data: (m,n) nparray of m data points (values castable to int)
    """
    # TODO: add score f'n parameter, default to empirical MI?  or too complicated?
    def MId(data, weights):
        """Estimate mutual information between all pairs of discrete variables"""
        m,n = data.shape
        d = data.max(0)+1
        MI = np.zeros((n,n))
        for i in range(n):
          for j in range(i+1,n):
            pij = empirical([[Var(i,d[i]),Var(j,d[j])]],data)[0]; pij+=1e-30; pij /= pij.sum()
            MI[i,j] = (pij*(pij/pij.sum([i])/pij.sum([j])).log()).sum()
            MI[j,i] = MI[i,j]
        return MI,None,None

    data = np.asarray(data)
    m,n = data.shape
    d = data.max(0)+1
    MI, _,_ = MId(data, weights)   
    from scipy.sparse.csgraph import minimum_spanning_tree as mst
    tree = mst(penalty-MI).tocoo();
    factors = [empirical([[Var(i,d[i])]], data)[0] for i in range(n)]
    for f in factors: f /= f.sum()
    for i,j,w in zip(tree.row,tree.col,tree.data):
        if w>0: continue
        (i,j)=(int(i),int(j)) if i<j else (int(j),int(i))
        fij = empirical([[Var(i,d[i]),Var(j,d[j])]], data)[0]; fij /= fij.sum()
        fij = fij / fij.sum([i]) / fij.sum([j])
        factors.append(fij)
    return GraphModel(factors)



################################################################################################
# "Vectorize" the model parameters (log factor values), in overcomplete exponential family form.
# "features" is a set of base factors used to determine the size & arrangement of the vector
# "factors" are the model factors to conver to the vector representation
# "theta" is the resulting log-vector

# TODO: update to accept either a factor list or a graphical model
# TODO: update to check for isLog flag
# TODO: add transform for data (x-configuration) to feature indicators?

def vectorize(factors, features, default=0.0):
  """Return a vectorization of the model with "factors", under the specified set of base features"""
  # TODO: better documentation
  model = GraphModel(features, copy=False)  # create model with references to feature factors
  idx = {};
  t = 0;
  for u in model.factors:
    idx[u] = slice(t,t+u.numel());  # save location of this factor's features in the full vector
    t += u.numel()
  theta = np.zeros((t,))+default;           # allocate storage for vectorization
  for f in factors:
    u = model.factorsWithAll(f.vars)[0]; # get smallest feature set that can contain this factor
    theta[idx[u]] += f.log().table.ravel(order=orderMethod);   # and add the factor to it
  return theta

def devectorize(theta, features, default=0.0, tolerance=0.0):
  """Return a list of factors from a vectorized version using the specified set of base features"""
  t = 0;
  factors = []
  for u in features:
    fnext = Factor(u.vars, theta[t:t+u.numel()]);
    t += u.numel();
    if (fnext-default).abs().sum() > tolerance:   # if any entries are different from the default,
      factors.append( fnext.expIP() );            #   add them as factors  (exp?)
  return factors

################################################################################################



