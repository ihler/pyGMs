"""
ising.py

Specialty graphical model class for Ising models (binary pairwise models)
Note: uses data definition Xi in {0,1} for compatibility with other graphmodel classes

Version 0.1.1 (2022-04-06)
(c) 2020 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import operator as operator
import numpy as np

inf = float('inf')

from pyGMs.factor import *
from pyGMs.graphmodel import *


from scipy.sparse import coo_matrix as coo
from scipy.sparse import csr_matrix as csr

from numpy import asarray as arr
from numpy import atleast_2d as twod

def toPM(x):
    return 2*x-1
def to01(x):
    return (x>0).astype(int)

class Ising(object):
  """Specialized graphical model class for representing Ising models (pairwise, binary models)
    Internal representation is as a sparse matrix, which is much more efficient than a factor list.
    NOTE: evaluations / data expect Xi = {0,1} to match GraphModel class
  """
  # TODO: add error checking, or expectation 0,1 to match normal GM?  Add valuePM, logValuePM, etc?
  # X : all variables are binary or single-state
  # factors: theta_i and L arrays
  # factorsByVar is L adjacency / non-zero

  isLog = False   # Note: unlike GraphModel, this does not affect the internal representation, only accessors!
  c = 0.                    # constant term
  h = np.array([]);         # single-variable terms (numpy vector)
  L = csr(np.array([[]]));  # pairwise terms (sparse array)
  dims = np.array([], dtype=int); # dimension of variables (2, or 1 if conditioned, 0 if eliminated)
  # TODO: keep track of "dimensions": 2 (normal), 0 (undefined var), 1 (elim var)
  # TODO: keep track of "known values" for conditioned model: +/- 1 or 0 / missing

  def __repr__(self):
    return "Ising model: {} vars, {} factors".format(self.nvar, self.nfactors)

  def __init__(self, factorList=None, copy=True, isLog=False):
    """Take in a list of factors and convert & store them in the internal format
      Can also accept a matrix of Ising parameters
    """
    if factorList is None:
      self.h = np.zeros(0); self.L=csr((0,0)); return;
    if not isinstance(factorList[0], Factor): # not a factor list => matrix?
      L = coo(factorList)
      LL = csr(factorList)
      n = L.shape[0]
      self.h = np.array([LL[i,i] for i in range(n)]);    # extract diagonal
      self.dims = np.array([2 for i in range(n)], dtype=int);   # all variables binary
      keep = (L.row != L.col)
      data,row,col = L.data[keep],L.row[keep],L.col[keep]
      #for j in np.where(L.row > L.col): row[j],col[j] = col[j],row[j]
      self.L = csr((data,(row,col)),shape=(n,n))   # keep in csr format
      self.L = .5*(self.L+self.L.T)  # force symmetric if not (TODO: detect zeros & overwrite?)
    else:
      n = np.max([np.max(f.vars.labels) for f in factorList if len(f.vars)])+1
      assert np.max([np.max(f.vars.dims()) for f in factorList]) <= 2, "Variables must be binary"
      assert np.max([f.nvar for f in factorList]) <= 2, "Factors must be pairwise"
      self.dims = np.zeros((n,), dtype=int)
      for f in factorList:
        for v in f.vars: self.dims[v] = v.states
      self.h = np.zeros(n);
      self.L = csr(([],([],[])),shape=(n,n));
      self.addFactors(factorList, isLog=isLog)
      


  def toLog(self): isLog=True; return self;
  def toExp(self): isLog=False; return self;
  def copy(self):
    """Return a (deep) copy of the Ising model"""
    import copy as pcopy
    return pcopy.deepcopy(self)

  def addFactors(self, flist, copy=True, isLog=False):
    """Add a list of (binary, pairwise) factors to the model; factors are converted to Ising parameters"""
    row = np.zeros(2*len(flist),dtype=int)-1; col=row.copy(); data=np.zeros(2*len(flist));
    for k,f in enumerate(flist):
      if not isLog: 
          if np.any(f.t<=0): f = f+1e-10;   # TODO: log nonzero tol
          f = f.log()
      if f.nvar == 1:
        Xi = f.vars[0]
        self.h[Xi] += .5*(f[1]-f[0])
        self.c += .5*(f[1]+f[0])
      else:
        Xi,Xj = f.vars[0],f.vars[1]
        row[2*k],col[2*k],data[2*k] = int(Xi),int(Xj), .25*(f[1,1]+f[0,0]-f[0,1]-f[1,0])
        row[2*k+1],col[2*k+1],data[2*k+1] = col[2*k],row[2*k],data[2*k]          
        #L[Xi,Xj] += .25*(f[1,1]+f[0,0]-f[0,1]-f[1,0])
        self.h[Xi] += .5*(f[1,0]-f[0,0])+data[2*k] #L[Xi,Xj]
        self.h[Xj] += .5*(f[0,1]-f[0,0])+data[2*k] #L[Xi,Xj]
        self.c += .25*(f[1,1]+f[1,0]+f[0,1]+f[0,0])
    self.L += csr((data[row>=0],(row[row>=0],col[row>=0])),shape=(self.nvar,self.nvar));

    

  def removeFactors(self,flist, isLog=False):
    """Remove a list of factors from the model
    
    >>> model.removeFactors(model.factorsWith([0]))    # remove all factors involving X0
    """
    # Currently: just divide out factors (add inverse factors) -- can't check if factor present? (minimal)
    # TODO: set entries to zero, then call self.L.eliminate_zeros()
    row = np.zeros(2*len(flist),dtype=int)-1; col=row.copy(); data=np.zeros(2*len(flist));
    for k,f in enumerate(flist):
      if not isLog: 
          if np.any(f.t==0): f = f+1e-30;   # TODO: log nonzero tol
          f = f.log()
      if f.nvar == 1:
        Xi = f.vars[0]
        self.h[Xi] -= .5*(f[1]-f[0])
        self.c -= .5*(f[1]+f[0])
      else:
        Xi,Xj = f.vars[0],f.vars[1]
        row[2*k],col[2*k],data[2*k] = int(Xi),int(Xj), .25*(f[1,1]+f[0,0]-f[0,1]-f[1,0])
        row[2*k+1],col[2*k+1],data[2*k+1] = col[2*k],row[2*k],data[2*k]          
        #L[Xi,Xj] += .25*(f[1,1]+f[0,0]-f[0,1]-f[1,0])
        self.h[Xi] -= .5*(f[1,0]-f[0,0])+data[2*k] #L[Xi,Xj]
        self.h[Xj] -= .5*(f[0,1]-f[0,0])+data[2*k] #L[Xi,Xj]
        self.c -= .25*(f[1,1]+f[1,0]+f[0,1]+f[0,0])
    self.L -= csr((data[row>=0],(row[row>=0],col[row>=0])),shape=(self.nvar,self.nvar));    
    #raise NotImplementedError();

  def makeMinimal(self): return;   # already minimal
  def makeCanonical(self): return; # already canonical

  def value(self,x,subset=None):
    return np.exp(self.logValue(x,subset))

  def logValue(self,x,subset=None):
    """Evaluate log F(x) for a configuration or data set
      x : (m,n) or (n,) array or dict : configuration(s) x to evaluate
    """
    if subset is not None: raise NotImplementedError()    # TODO: use L[subset,subset]?
    if isinstance(x,dict): x = toPM(np.array([[x[i] for i in range(self.nvar)]]))
    else: x = toPM(twod(arr(x)))
    r = self.L.dot(x.T)/2
    if len(x.shape)==2: r += self.h.reshape(-1,1);
    else:               r += self.h;
    return (x.T*r).sum(0) + self.c 

  def isBinary(self): return True  # Check whether the model is binary (all variables binary)
  def isPairwise(self): return True # Check whether the model is pairwise (all factors pairwise)
  def isCSP(self): return False
  def isBN(self, tol=1e-6): return False # Check whether the model is a valid Bayes Net

  @property
  def vars(self):
    """List of variables in the graphical model; equals model.X"""
    return [Var(i,self.dims[i]) for i in range(self.nvar)]  # TODO: use stored state info (=1 sometimes)

  @property
  def X(self): return self.vars

  def var(self,i):   # TODO: change to property to access (read only?) X?
    """Return a variable object (with # states) for id 'i'; equals model.X[i]"""
    return Var(i,self.dims[i])  

  @property
  def nvar(self):
    """The number of variables ( = largest variable id) in the model"""
    return self.h.shape[0]

  @property
  def nfactors(self):
    """The number of factors in the model"""
    return self.L.nnz

  @property
  def factors(self):
    """Return a list of factors (converted to full tables)"""
    X = [Var(i,2) for i in range(self.nvar)]
    factors = [Factor([],np.exp(self.c))] 
    # TODO: exclude if zero? or exclude if inf/-inf, or if in "assigned", or?
    factors = factors + [Factor([X[i]],[-th,th]).exp() for i,th in enumerate(self.h) if self.dims[i]>1]
    L = coo(self.L)
    factors = factors + [Factor([X[i],X[j]],[[th,-th],[-th,th]]).exp() for i,j,th in zip(L.row,L.col,L.data) if i<j]
    return factors
    # TODO: should we exponentiate if isLog not True? 

  def factorsWith(self,v,copy=True): 
    """Return a list of factors (converted to tables) in the model that contain the variable 'v'"""
    Lv = self.L.getrow(v).tocoo();
    factors = [Factor([Var(int(v),2)],[-th,th]).exp() for th in [self.h[v]] if self.dims[v]>1]
    factors = factors + [Factor([Var(int(v),2),Var(int(j),2)],[[th,-th],[-th,th]]).exp() for j,th in zip(Lv.col,Lv.data)]
    return factors

  def factorsWithAny(self,vs): 
    """Return a list of factors (converted to tables) in the model that contain any of the variables in 'vs'"""
    factors = []
    for v in vs:
      factors += [Factor([Var(int(v),2)],[-th,th]).exp() for th in [self.h[v]] if self.dims[v]>1]
      for u in self.markovBlanket(v):
        if u not in vs or v < u:
          factors += [Factor([Var(int(v),2),Var(int(u),2)],[[th,-th],[-th,th]]).exp() for th in [self.L[v,u]] if th!=0] 
    return factors

  def markovBlanket(self,v): 
    """Return the Markov Blanket (list of neighbors) of a given variable in the model"""
    return VarSet([Var(int(i),2) for i in self.L.getrow(int(v)).nonzero()[1]])
    #return self.L.getrow(int(v)).nonzero()[1].astype(int)

  def degree(self, v=None):
    """Return the degree (number of neighbors) of one or more variables (default: all)"""
    if v is None: return arr((self.L!=0).sum(1)).reshape(-1);
    else:         return (self.L[v,:]!=0).sum();

  def __asFactor(i,j=None):
    # TODO: fix up to be used in above functions 
    if j is None: return Factor([Var(int(i),2)],[-self.h[i],self.h[i]]).exp()
    th = self.L[i,j]
    return Factor([Var(int(i),2),Var(int(j),2)],[[th,-th],[-th,th]]).exp()

  def condition2(self, vs, xs): 
    """Create a clamped (or "sliced") factor using partial conditioning (list+list version)
       e.g., F.condition2([0,2],[a,b]) = f(x1,x3) = F(x_0=a,x_1,x_2=b,x_3)
    """
    # TODO: "remove" variable by setting states = 1, and saving "known value"?
    vs = np.array([int(v) for v in vs]); pi = np.argsort(vs);
    xs = toPM(np.asarray(xs)[pi])
    keep = [i for i in range(self.nvar) if self.vars[i] not in vs]
    drop = [i for i in range(self.nvar) if self.vars[i] in vs]
    self.c += xs.dot( self.L[np.ix_(drop,drop)].dot(xs)/2 + self.h[drop])  # DONE: check shape
    self.h[keep] += self.L[np.ix_(keep,drop)].dot(xs); self.h[drop] = 0;
    self.L[np.ix_(keep,drop)] = 0; self.L[np.ix_(drop,keep)] = 0; self.L[np.ix_(drop,drop)] = 0;
    self.L.eliminate_zeros();

  def condition(self, evidence):
    """Create a clamped (or "sliced") factor using partial conditioning (dict version)
       e.g., F.condition({0:a,2:b}) = f(x1,x3) = F(x_0=a,x_1,x_2=b,x_3)
    """
    return self.condition2( [x for x in evidence], [evidence[x] for x in evidence] )

  def slice2(self, vs=None,xs=None):
    """Create a clamped (or "sliced") factor using partial conditioning (list+list version)"""
    return self.condition2(vs,xs)

  def slice(self, evidence={}):
    """Create a clamped (or "sliced") factor using partial conditioning (dict version)"""
    return self.condition(evidence)
    
    

  def __TODO_eliminate(self, elimVars, elimOp):
    # TODO: awkward way to define this; convert to more direct implementation?
    for v in elimVars:
      if len(self.markovBlanket(v)) > 2: raise ValueError("Cannot eliminate {} with {} (>2) neighbors".format(v,len(self.markovBlanket(v))))
      flist = self.factorsWith(v)
      gm_model = GraphModel(flist); print(gm_model); gm_model.eliminate([v],elimOp)
      fnew = gm_model.factors[0]
      self.removeFactors(flist);  # doesn't quite work? numerical roundoff issues?
      self.L[v,:] = 0; self.L[:,v] = 0; self.h[v] = 0;  # TODO: better to mark as removed? how?
      self.addFactors([fnew])
    # TODO:  "remove" variable by setting states = 0?  "known value" = 0?

  def joint(self): 
    """Return the (possibly intractably large) joint probability table for the Ising model"""
    return GraphModel(self.factors).joint()

  def connectedComponents(self):
    """Find the connected components of the model's Markov graph.  Returns a list of sets of variables."""
    components = []
    X = set(self.X)
    while X:
        Xi = X.pop()
        if Xi.states <= 1: continue   # don't include missing or assigned variables 
        group = {Xi}                  # start a new group with this variable
        queue = [Xi]                  # do DFS on the graph from Xi to find its connected component:
        while queue:
            n = queue.pop()
            nbrs = self.markovBlanket(n)   # get all connected variables
            nbrs.difference_update(group)  # remove any we've already seen
            X.difference_update(nbrs)      # remove new ones from unexplored variable list
            group.update(nbrs)             # add them to this connected component
            queue.extend(nbrs)             # and continue exploring from them in DFS order
        components.append(group)
    return components

  def nxMarkovGraph(self, all_vars=False):
    """Return a networkx object representing the Markov graph of the Ising model"""
    import networkx as nx
    return nx.from_scipy_sparse_matrix(self.L!=0)
    
  def pseudolikelihood(self, data):
    """Compute the pseudo (log) likelihood, \sum_i \sum_j \log p(x^{(j)}_i | x^{(j)}_{\neg i})
       data : (m,n) or (n,) array or dict of the values xi; values in {0,1}
    """
    if isinstance(data,dict): data = toPM(np.array([[data[i] for i in range(self.nvar)]]))
    else: data = toPM(twod(data));              # interface glue: convert {0,1} to {-1,+1}
    r = self.L.dot(data.T)
    r += self.h.reshape(-1,1) if len(data.shape)==2 else self.h
    lnp = -np.log(1+np.exp(-2*data.T*r))    # ln p(x_i^(s)|x_{-i}^(s)) for all vars i, samples s
    return lnp.sum(axis=0)                # sum over i => pseudo-log-likelihood of each x^(s)


### TO ADD:
# Likelihood eval (requires LPF estimate)
# LBP, NMF Ising optimized versions?
# local logreg estimates combined: average, weighted avg, min-nbrs, etc
#   Lasso-And, Lasso-Or (see M&B or Banerjee08; W&J for ising version)
# Structure via SDP (Banerjee08)
# Structure via independence tests (ref?)
# L1-regularized pseudolikelihood optimization? 
# re-fit functions (fix nonzero structure of L): pseudolikelihood SGD; IPF for LL; moments pij/pi/pj
# screening: find blockwise independence given regularization lambda 
#    https://papers.nips.cc/paper/6674-a-screening-rule-for-l1-regularized-ising-model-estimation.pdf


  def fit_chowliu(self, data, penalty=0, weights=None):
    """Select a maximum likelihood tree-structured graph & parameters
      data: (n,m) nparray of m data points; values {0,1}
    """
    # TODO: add score f'n parameter, default to empirical MI?  or too complicated?
    def MI2(data, weights):
        """Estimate mutual information between all pairs of *binary* {0,1} variables"""
        pi = np.average(data.astype(float),axis=1,weights=weights)[np.newaxis,:]
        pij = np.cov(data,ddof=0,aweights=weights) + (pi.T.dot(pi));
        p = np.stack( (pij, pi-pij, pi.T-pij, 1+pij-pi-pi.T), axis=2)
        p2 = pi.T.dot(pi)
        q = np.stack( (p2,pi-p2,pi.T-p2,1+p2-pi-pi.T), axis=2)
        MI = (p*(np.log(p+1e-10)-np.log(q+1e-10))).sum(axis=2)
        return MI,pij,pi[0]
        
    n,m = data.shape
    #MI, pij,pi = MI2(to01(data), weights)
    MI, pij,pi = MI2(data, weights)       # data should be 0/1, not -1/+1
    from scipy.sparse.csgraph import minimum_spanning_tree as mst
    tree = mst(penalty-MI).tocoo();
    factors = [Factor([Var(i,2)], [1-pi[i],pi[i]]) for i in range(n)]
    for i,j,w in zip(tree.row,tree.col,tree.data):
        if w>0: continue
        (i,j)=(int(i),int(j)) if i<j else (int(j),int(i))
        tij = [1+pij[i,j]-pi[i]-pi[j], pi[i]-pij[i,j], pi[j]-pij[i,j], pij[i,j]]
        fij = Factor([Var(i,2),Var(j,2)],tij);
        fij = fij / fij.sum([i]) / fij.sum([j])
        factors.append(fij)
    self.__init__(factors)


# TODO: FIX INTERFACE ISSUES / CONSISTENCY
# TODO: BP/IPF parameter estimates (use MI2 function); rewrite fitCL to use this?


def __pll(L,h,x, L2=0):
    """Evaluate the pseudo(log)likelihood of an Ising model (L,h).  X in {-1,+1}."""
    # TODO: still expects (n,m) shaped array
    if len(x.shape)>1: h = h.reshape(-1,1);
    pll = -np.log(1+np.exp(-2*x*(L.dot(x)+h))).sum(0)
    if L2>0: pll += L2*(L**2).sum()
    return pll


def __dpll(L,h,x, L2=0):
    """Evaluate the pseudo(log)likelihood gradient of an Ising model (L,h).  X in {-1,+1}."""
    # TODO: still expects (n,m) shaped array
    if len(x.shape)>1: h = h.reshape(-1,1);
    p = 1./(1+np.exp(2*x*(L.dot(x)+h))) # compute p(x^s_i|x^s_!i) for all i,s
    dh = 2*p*x
    if len(x.shape)>1: dh = dh.mean(1);  # average over data if x[i] are vectors
    dL = L.tocoo()
    for k in range(dL.nnz):
        i,j = dL.row[k],dL.col[k]
        dL.data[k] = 2*np.mean((p[i]+p[j])*(x[i]*x[j]))  # avg over s if needed
    return dL.tocsr(),dh




def refit_pll_sgd(model,data, initStep=.01, maxIter=1000, verbose=False):
    """Fit a fixed graph structure to optimize pseudo-log-likelihood (uses basic SGD)
      model : an Ising model to re-fit (will keep edge structure)
      data  : (m,n) array of m data points with n variables (with values {0,1})
    """
    data = toPM(data.T);   # TODO: internal (n,m) shape
    last = 0
    for it in range(maxIter):
        stepi = 10*initStep / (10+it)
        dLL,dHH = __dpll(model.L,model.h,data)
        model.L += stepi*dLL
        model.h += stepi*dHH
        if (verbose and it >= 2*last): last=it; print(it, ": ", __pll(model.L,model.h,data).mean()); 


def refit_pll_opt(model,data):
    """Fit a fixed graph structure to optimize pseudo-log-likelihood (uses scipy optimize)
      model : an Ising model to re-fit (will keep edge structure)
      data  : (m,n) array of m data points with n variables (with values {0,1})
    """
    data = toPM(data.T);   # TODO: internal (n,m) shape
    import scipy.optimize
    from scipy.sparse import triu
    def to_vector(L,h):
        return np.hstack((h,triu(L,k=1).tocoo().data))
    def from_vector(x):
        h = x[:len(model.h)];
        tmp = triu(model.L,k=1).tocoo();
        L = csr((x[len(model.h):],(tmp.row,tmp.col)),shape=model.L.shape)
        return L+L.T,h
    def f0(x0):
        L,h = from_vector(x0)
        return -__pll(L,h,data).mean()
    def jac(x0):
        L,h = from_vector(x0)
        return -to_vector(*__dpll(L,h,data))

    x0 = to_vector(model.L,model.h)
    res = scipy.optimize.minimize(f0,x0, method='BFGS',jac=jac)
    #print("Success? ",res.success)
    model.L,model.h = from_vector(res.x)
    return res

# default: use scipy optimize?
refit_pll = refit_pll_opt;


# TODO: add "symmetrize" function:  (see ICML'11?)
#   basic: L = .5*(L+L.T)
#   min-degree:  L[i,j] = L[j,i] if nnz(L[i,:]) < nnz(L[j,:])
#   weighted: ...


def fit_logregL1(data, C=.01):
    """Estimate an Ising model using penalized logistic regression
      data: (n,m) array of m data points in {0,1}
      C: float, sparsity penalty (smaller = sparser graph)
    """
    from sklearn.linear_model import LogisticRegression
    m,n = data.shape
  
    # TODO: just build (sparse) L directly & construct with it
    # for each Xi, estimate the neighborhood of Xi using L1-reg logistic regression:
    nbrs,th_ij,th_i = [None]*n, [None]*n, np.zeros((n,))
    Xtr, Xtmp = toPM(data.T), toPM(data.T)  # make two copies so we can modify; TODO: internal (n,m) shape
    for i in range(n):  
        Xtmp[i,:] = 0.        # remove ourselves
        lr = LogisticRegression(penalty='l1',C=C,solver='liblinear').fit(Xtmp.T,Xtr[i,:])
        nbrs[i] = np.where(np.abs(lr.coef_) > 1e-6)[1]
        th_ij[i]= lr.coef_[0,nbrs[i]]/2.
        th_i[i] = lr.intercept_/2.
        Xtmp[i,:] = Xtr[i,:]; # & restore after
    
    # Collect single-variable factors
    factors = [Factor(Var(i,2),[-t,t]).exp() for i,t in enumerate(th_i)]

    # Collect non-zero pairwise factors
    for i in range(n):
        for jj,j in enumerate(nbrs[i]):
            # TODO: FIX: double counts edges? use different combination methods?
            scope = [Var(i,2),Var(int(j),2)]
            t = th_ij[i][jj]
            factors.append( Factor(scope, [[t,-t],[-t,t]]).exp() )
    
    # Build a model from the factors
    return Ising(factors)


def fit_mweight(data, C=1., threshold=1e-4, learning_rate=None):
    """Estimate an Ising model using multiplicative weights (Klivans & Meca '17)
       data: (m,n) array of m data points in {0,1}
       C: float, sparsity bound (smaller = sparser graph)
       threshold: float, cutoff for making parameters exactly zero (larger = sparser graph)
       learning_rate: float, (1-epsilon) learning rate for "Hedge" multiplicative weights
    """
    dataPM = toPM(data);
    m,n = dataPM.shape;
    if learning_rate is None: learning_rate = 1-np.sqrt(np.log(n)/m);

    L,h = np.zeros((n,n)), np.zeros((n,))    # initialize parameters (dense) & weights
    eye = list(range(n));
    Wp = np.ones((n,n))/(2*(n-1)); Wp[eye,eye] = 0; Wm = np.copy(Wp);
    Hp = np.ones((n,))/2; Hm = np.copy(Hp);

    for i,xpm in enumerate(dataPM):
        phat = 1./(1.+np.exp(2.*L.dot(xpm) + h))
        ell_H = (phat - (1.-xpm)/2.);
        ell_W = np.outer( ell_H , xpm);
        Wp *= learning_rate**(-ell_W); Wm *= learning_rate**(ell_W);
        Hp *= learning_rate**(-ell_H); Hm *= learning_rate**(ell_H);
        L = C/(Wp.sum(1)+Wm.sum(1))[:,np.newaxis]*(Wp-Wm)
        h = C/(Hp+Hm)*(Hp-Hm)
    
    L = .5*(L+L.T);
    L[np.abs(L)<threshold] = 0;
    L[eye,eye] = h;
    return Ising(L);


# TODO change to "threshold" ; scale max edges by # nodes? 
def fit_threshold(data, rho_cutoff=0., maxedges=None, diag=1e-6):
    """Estimate an Ising model using a (trivial) thresholded inverse covariance estimate.
    data: (m,n) array of m data points in {0,1}
    rho_cutoff: minimum value of a non-zero pairwise conditional corrrelation (larger = sparser)
    maxedges: maximum number of edges to keep (smaller = sparser)
    """
    from scipy.linalg import inv as scipy_inv
    m,n = data.shape
    sig = np.cov(data.T,ddof=0,aweights=None) + diag*np.eye(n);
    J   = -scipy_inv(sig);  J = .5*(J+J.T);            # symmetrize just in case
    Jdiag = J[range(n),range(n)]; J[range(n),range(n)] = 0.   # zero diagonal for threshold op
    if maxedges is not None:
      if maxedges > J.size: rho_cutoff = 0.
      else: rho_cutoff = max(rho_cutoff, -np.sort(np.abs(J).reshape(-1))[2*maxedges])
    J[ np.abs(J) <= rho_cutoff ] = 0.
    J[range(n),range(n)] = Jdiag;                             # restore diagonal
    J[range(n),range(n)] -= J.dot(np.mean(data.T,1))          # add singleton terms (TODO: CHECK)
    return Ising(J)

def fit_chowliu(data, penalty=0, weights=None):
    """Estimate an Ising model using Chow-Liu's max likelihood tree structure & parameters
      data: (m,n) nparray of m data points; values {0,1}
      penalty: non-negative penalty on the MI (may give a disconnected / forest graph)
    """
    # TODO: add score f'n parameter, default to empirical MI?  or too complicated?
    def MI2(data, weights, eps=1e-10):
        """Estimate mutual information between all pairs of *binary* {0,1} variables"""
        # TODO: expects (n,m) shape data
        pi = np.average(data.astype(float),axis=1,weights=weights)[np.newaxis,:]
        pij = np.cov(data,ddof=0,aweights=weights) + (pi.T.dot(pi));
        p = np.stack( (pij, pi-pij, pi.T-pij, 1+pij-pi-pi.T), axis=2)
        p2 = pi.T.dot(pi)
        q = np.stack( (p2,pi-p2,pi.T-p2,1+p2-pi-pi.T), axis=2)
        MI = (p*(np.log(p+eps)-np.log(q+eps))).sum(axis=2)
        return MI,pij,pi[0]
        
    m,n = data.shape
    MI, pij,pi = MI2(data.T, weights)       # data should be 0/1, not -1/+1
    from scipy.sparse.csgraph import minimum_spanning_tree as mst
    tree = mst(penalty-MI).tocoo();
    factors = [Factor([Var(i,2)], [1-pi[i],pi[i]]) for i in range(n)]
    for i,j,w in zip(tree.row,tree.col,tree.data):
        if w>0: continue
        (i,j)=(int(i),int(j)) if i<j else (int(j),int(i))
        tij = [1+pij[i,j]-pi[i]-pi[j], pi[i]-pij[i,j], pi[j]-pij[i,j], pij[i,j]]
        fij = Factor([Var(i,2),Var(j,2)],tij);
        fij = fij / fij.sum([i]) / fij.sum([j])
        factors.append(fij)
    return Ising(factors)

def fit_greedy(data, nnbr=10, threshold=0.05, refit=refit_pll):
    """Estimate an Ising model using Bresler's greedy edge selection approach
      data: (m,n) nparray of m data points; values {0,1}
      nnbr: maximum number of neighbors to allow for any node
      threshold: expected variation threshold to declare an edge (in [0,1])
      refit: function of (model,data) to optimize parameter values given graph structure
    """
    m,n = data.shape;
    L = np.zeros((n,n))    # initialize parameters
    scores = np.zeros(n)   
    data = data.T.astype(int)    # TODO: fix internal transpose
    for i in range(n):
        Ni = []
        while (len(Ni)<nnbr):
            Vi = (0*data[i,:] + sum(data[j,:]*(2**jj) for jj,j in enumerate(Ni))).astype(int)
            Vsz = int(Vi.max()+1)
            for j in range(n):
                if j==i or j in Ni: scores[j]=0.; continue
                pIJV = Factor( [Var(0,2),Var(1,2),Var(2,Vsz)] , 0.)
                # pIJV[data[i,:],data[j,:],Vi] += 1.  # Test??
                for k in range(m): pIJV[data[i,k],data[j,k],Vi[k]] += 1.
                pV = pIJV.marginal([2]); pV /= (pV.sum()+1e-20);
                pIJV /= (pIJV.sum([0])+1e-20)
                scores[j] = ((pIJV.condition({0:1,1:1})-pIJV.condition({0:1,1:0})).abs()*pV).sum()
            jmax = int(np.argmax(scores))
            if scores[jmax] < threshold: break
            Ni.append(jmax)
        # TODO: prune back each list?
        #print(i," : ",Ni)
        L[i,Ni] = 1.
    L = L*L.T  # "and" connectivity: keep only if edges (i,j) and (j,i) present?
    model = Ising(L);
    refit(model,data.T)
    return model


def __Bethe(ising,R,mu,bel=None):
    if bel is None: bel = 1./(1+np.exp(-2.*(arr(mu.sum(0)).reshape(-1)+ising.h)))
    RT = R.T.tocsr();
    assert R.has_canonical_format and RT.has_canonical_format and ising.L.has_canonical_format, "CSRs must be in canonical format"
    B = np.vstack( (ising.L.data-R.data-RT.data, -ising.L.data-R.data+RT.data, -ising.L.data+R.data-RT.data, ising.L.data+R.data+RT.data) );
    B = np.exp(B); B /= B.sum(0,keepdims=True);  # pairwise beliefs

    E2 = B.T.dot([1.,-1.,-1.,1.]).dot(ising.L.data);  # L.data order same as R,RT (!)
    H2 = (B*np.log(B)).sum()                 # full vector: H2 = (B*np.log(B)).sum(1)
    E1 = (2.*bel-1).dot(ising.h);            # "" : E1 = (2.*bel-1) * ising.h;
    H1 = (bel*np.log(bel) + (1-bel)*np.log(1-bel));
    return .5*(E2-H2) + E1 - (H1.dot(1.-ising.degree())) + ising.c;


def LBP(ising, maxIter=100, verbose=False):
    """Run loopy belief propagation (specialized for Ising models)
       lnZ, bel = LBP(ising, maxIter, verbose)
       lnZ : float, estimate of the log partition function
       bel : vector, bel[i] = estimated marginal probability that Xi = +1
    """
    # TODO: pass requested beliefs (like JT?), or "single", "factors", etc.
    assert isinstance(ising,Ising), "Model must be an Ising model for this version to work"
    R = ising.L.tocoo(); row = R.row; col = R.col;
    mu = csr(([],([],[])),shape=ising.L.shape)
    L_tanh = ising.L.tanh();
    for it in range(maxIter):
        mu_sum = arr(mu.sum(0)).reshape(-1);
        #R = csr( (ising.h[row]+mu_sum[row], (row,col)), shape=ising.L.shape) - mu.T
        R = csr( (ising.h[row]+mu_sum[row]-arr(mu[col,row]).reshape(-1), (row,col)), shape=ising.L.shape);
        mu = (L_tanh.multiply(R.tanh())).arctanh()
        if verbose: print("Iter "+str(it)+": "+str(__Bethe(ising,R,mu)));

    R = csr( (ising.h[row]+mu_sum[row]-arr(mu[col,row]).reshape(-1), (row,col)), shape=ising.L.shape);
    bel = 1./(1+np.exp(-2.*(arr(mu.sum(0)).reshape(-1)+ising.h)))
    lnZ = __Bethe(ising,R,mu,bel)
    return lnZ, bel




