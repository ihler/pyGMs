"""
graphmodel.py

Defines a graphical model container class for reasoning about graphical models

Version 0.3.1 (2025-08-15)
(c) 2015-2025 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import operator as operator
import numpy as np
from sortedcontainers import SortedSet;
from sortedcontainers import SortedListWithKey;

inf = float('inf')

from pyGMs.factor import *


def _is2D(D):
  try: next(iter(next(iter(D))))
  except: return False
  return True


# Make a simple sorted set of factors, ordered by clique size, then lexicographical by scope
def factorSet(it=None): return SortedSet(iterable=it,key=lambda f:'{:04d}.'.format(f.nvar)+str(f.vars)[1:-1])
# So, fs = factorSet([list]) will make sure:
#   fs[0] is a minimal factor (smallest # of variables) and fs[-1] is a maximal factor (largest # of variables)

class GraphModel(object):
  """A basic graphical model class; represents a collection of factors.

  Example:

  >>> flist = readUai('myfile.uai')  # read a list of factors from a UAI format file
  >>> model = GraphModel(flist)      # makes a copy of the factors for manipulation

  The model may be stored in an exponential, product of factors form:  f(X) = \\prod_r f_r(X_r),
  or in a log-probability, sum of factors form:  \theta(X) = \\sum_r \theta_r(X_r).
  
  Various accessor functions enable finding factors that depend on one or more variables, variables that
  share one or more factors (their Markov blanket), manipulations to the graph (such as eliminating one
  or more variables), and visualization (through networkx).
  """

  X            = []            # variables model defined over
  factors      = factorSet()   # collection of factors that make up the model, sorted by scope size, lex
  factorsByVar = []            # factor lookup by variable:  var -> factor list 
 
  #TODO: useful stuff for algorithm objects interacting with the graph
  lock         = False         # are factors "locked", or are reparameterization changes OK?
  ## TODO: change to structure lock & parameter lock, separate?
  sig          = 0;            # "signature" of factor cliques; changes if model structure (cliques) changes
 
  isLog = False                # flag: additive (log probability) model vs multiplicative (probability) model 


  def __repr__(self):
    # TODO: add any details?
    return "Graphical model: {} vars, {} factors".format(self.nvar,self.nfactors)

  def __init__(self, factorList=None, copy=True, isLog=False):   
    """Create a graphical model object from a factor list.  Maintains local copies of the factors.

    Args:
       factorList (list): A list of factors defining the graphical model
       copy (bool): Whether the model should make its own internal copy of all factors; default True
       isLog (bool): Whether the factors are interpreted as log-probability factors; default False
    """
    self.X = []
    self.factors = factorSet()
    self.factorsByVar = []
    self.lock = False
    self.sig = 0
    self.isLog = isLog
    if not (factorList is None): self.addFactors(factorList, copy=copy)
  
  ###################################################
  ## Basic accessors: what variables are in the model?

  @property
  def vars(self):
    """List of variables in the graphical model; equals model.X"""
    return self.X

  @property
  def vars_nontrivial(self):
    """List of "nontrivial" variables (more than one state) in the graphical model"""
    return [x for x in self.X if x.states>1]

  def var(self,i):   # TODO: change to property to access (read only?) X?
    """Return a variable object (with # states) for id 'i'; equals model.X[i]"""
    return self.X[i]

  @property
  def nvar(self):
    """The number of variables ( = largest variable id) in the model"""
    return len(self.X)

  @property
  def nfactors(self): 
    """The number of factors in the model"""
    return len(self.factors)

  ###################################################
  ## (Partial) configurations of the model variables can be represented as dicts or tuples:

  def as_tuple(self, config, X=None):
    """Convert a (partial) configuration represented as a dict into a tuple.
      Variables in self.X missing from config are represented as NaN.
    """
    if X is None: X = self.X
    return tuple(config[x] if x in config else np.nan for x in X)

  def as_dict(self, config, X=None):
    """Convert a (partial) configuration represented as a tuple into a dict.
      Config entries specified by NaN are excluded.
    """
    if X is None: X=self.X
    return {x:config[x] for x in X if not np.isnan(config[x])}

  ###################################################
  ## Basic model transforms. "Log" model => additive cliques, "Exp" model => multiplicative cliques

  def toLog(self):
    """Convert internal factors to log form (if not already).  May use 'isLog' to check."""
    if not self.isLog:
      for f in self.factors: f.logIP()
      self.isLog = True
    return self

  def toExp(self):
    """Convert internal factors to exp form (product of probabilities) if not.  May use 'isLog' to check."""
    if self.isLog:
      for f in self.factors: f.expIP()
      self.isLog = False
    return self
 
  def copy(self):
    """Return a (deep) copy of the graphical model"""
    import copy as pcopy
    return pcopy.deepcopy(self)

  def makePositive(self,eps=1e-8):
    """Force factors / model to be positive (non-zero)"""
    if self.isLog: eps = np.log(eps)
    for f in self.factors:
      f.table = np.maximum( f.table, eps )

  def value(self,x,subset=None):
    """Evaluate F(x) = \\prod_r f_r(x_r) for some (full) configuration x.
         If optional subset != None, uses *only* the factors in the Markov blanket of subset.
    """
    if not _is2D(x): x=[x]
    factors = self.factors if subset==None else self.factorsWithAny(subset)
    if self.isLog: return np.exp( np.sum( [[ f.valueMap(xx) for f in factors ] for xx in x] ,1) )
    else:          return np.prod( [[ f.valueMap(xx) for f in factors ] for xx in x] ,1)

  def logValue(self,x,subset=None): 
    """Evaluate log F(x) = \\sum_r log f_r(x_r) for some (full) configuration x.
         If optional subset != None, uses *only* the factors in the Markov blanket of subset.
    """
    if not _is2D(x): x=[x]
    factors = self.factors if subset==None else self.factorsWithAny(subset)
    if self.isLog: return np.sum( [[ f.valueMap(xx) for f in factors ] for xx in x] ,1)
    else:          return np.sum( [[ np.log(f.valueMap(xx)) for f in factors] for xx in x] ,1)

  ###################################################
  ## Modifying graph structure & cliques

  def addFactors(self,flist,copy=True):
    """Add a list of factors to the model; factors are copied locally unless copy = False"""
    import copy as pcopy
    if (copy): flist = pcopy.deepcopy(flist)  # create new factor copies that this model "owns"
    self.factors.update(flist);   # add factor to set of all factors, and to by-variable indexing
    for f in flist:
      # TODO: if constant do something else?  (Currently just leave it in factors list)
      for v in reversed(f.vars):
        if (v.label >= len(self.X)): 
          self.X.extend( [Var(i,1) for i in range(len(self.X),v.label+1) ] )
          self.factorsByVar.extend( [ factorSet() for i in range(len(self.factorsByVar),v.label+1) ] )
        if self.X[v].states <= 1: self.X[v] = v   # copy variable info if undefined, then check:
        if self.X[v].states != v.states: raise ValueError('Incorrect # of states',v,self.X[v])
        self.factorsByVar[v].add(f)
    self.sig = self.__clique_signature()  # track graph clique signature

  def removeFactors(self,flist):
    """Remove a list of factors from the model
    
    >>> model.removeFactors(model.factorsWith([0]))    # remove all factors involving X0
    """
    self.factors.difference_update(flist)
    for f in flist:
      for v in f.vars:
        self.factorsByVar[v].discard(f)
    self.sig = self.__clique_signature()  # track graph clique signature

  def makeCanonical(self):
    """Add/merge factors to make a canonical factor graph: singleton factors plus maximal cliques"""
    for f in self.factors:
      fs = self.factorsWithAll(f.vars)
      if (f.nvar > 1) & (fs[-1] != f):
        if self.isLog: fs[-1] += f
        else:          fs[-1] *= f
        self.removeFactors([f])
    self.sig = self.__clique_signature()  # track graph clique signature
        
  def makeMinimal(self):
    """Merge factors to make a minimal factor graph: retain only factors over maximal cliques"""
    to_remove = []
    for f in self.factors:
      fs = self.factorsWithAll(f.vars)
      largest = fs[-1]
      if (fs[-1] != f):
        if self.isLog: largest += f
        else:          largest *= f
        to_remove.append(f)
    self.removeFactors(to_remove)
    self.sig = self.__clique_signature()  # track graph clique signature

  def __clique_signature(self):
      """Generate hash value based on the graph cliques. Used to track changes in graph structure."""
      cliquestr = ",".join( (str(f.vars) for f in self.factors) )
      return hash(cliquestr)

  ###################################################
  ## Querying graph structure & cliques; special classes of models

  def factorsWith(self,v,copy=True):
    """Get the list of all factors that include variable v"""
    return self.factorsByVar[v].copy() if copy else self.factorsByVar[v]

  def factorsWithAny(self,vs):
    """Get the list of all factors that include any variables in the list vs"""
    flist = factorSet()
    for v in vs: flist.update( self.factorsByVar[v] )
    return flist

  def factorsWithAll(self,vs):
    """Get the list of all factors that include all variables in the list vs"""
    if (len(vs)==0): return self.factors.copy()
    flist = self.factorsByVar[vs[0]].copy()
    for v in vs: flist.intersection_update(self.factorsByVar[v])
    return flist
  
  def markovBlanket(self,v):
    """Get the Markov blanket of variable v (all variables involved in a factor with v)"""
    vs = VarSet()
    for f in self.factorsByVar[v]: vs |= f.vars
    vs -= [v]
    return vs
 
  def isBinary(self):   
    """Check whether the graphical model is binary (all variables <= 2 states)"""
    return all( [x.states <= 2 for x in self.X] )

  def isPairwise(self): 
    """Check whether the graphical model is pairwise (has maximum scope size 2)"""
    return all( [len(f.vars)<= 2 for f in self.factors] )

  def isCSP(self): 
    """Check whether the graphical model is a valid CSP (all zeros or ones)"""
    if self.isLog: isTableCSP = lambda t : all( (t==-np.inf) | (t==0.) ) 
    else:          isTableCSP = lambda t : all( (t==0.) | (t==1.) )   
    return all( [isTableCSP(f.table) for f in self.factors] ) 

  def isBN(self, tol=1e-6):  # Check whether the model is a valid Bayes Net
    """Check whether the graphical model is a valid Bayes net (one CPT per variable) """
    topo_order = bnOrder(self)                              # TODO: allow user-provided order and check?
    if topo_order is None: return False
    # Now check to make sure each factor is a CPT for its last variable
    pri = np.zeros((len(topo_order),))-1
    pri[topo_order] = np.arange(len(topo_order))
    found  = [ 1 if x.states <= 1 else 0 for x in self.X ]  # track which variables have CPTs
    for f in self.factors:
      X = f.vars[ np.argmax([pri[x] for x in f.vars]) ]     # which is the last variable in this factor?
      found[X] = 1;
      tmp = f.sum([X]) - 1.0                         # check that each row sums to 1.0; TODO: assumes product semantics
      try:
        tmp = tmp.absIP().max()
      except:
        tmp = abs(tmp)
      if tmp > tol: return False                     # fail if not a CPT for X
    if np.prod(found)==0: return False               # fail if missing some variable's CPT
    return True
  

  ###################################################
  ## Graph transformations: conditioning on evidence, marginalizing variables

  def condition(self, evidence):
    """Condition (clamp) the graphical model on a partial configuration (dict) {Xi:xi, Xj:xj, ...}"""
    # TODO: optionally, ensure re-added factor is maximal, or modify an existing one (fWithAll(vs)[-1]?)
    if len(evidence)==0: return
    for v,x in evidence.items():
      constant = 0.0
      for f in self.factorsWith(v):
        self.removeFactors([f])
        fc = f.condition({v:x})                       
        if (fc.nvar == 0):
          if self.isLog: constant += fc[0]            # if it's now a constant, just pull it out 
          else:          constant += np.log(fc[0])
        else: 
          self.addFactors([fc],copy=False)            # otherwise add the factor back to the model
      # add delta f'n factor for each variable, and distribute any constant value into this factor
      if self.isLog: f = Factor([self.X[v]],-inf); f[x] = constant
      else:          f = Factor([self.X[v]],0.0); f[x] = np.exp(constant)
      self.addFactors([f],copy=False)

  def condition2(self, vs, xs):
    """Condition (clamp) the graphical model on the partial configuration vs=xs (may be lists or tuples)"""
    self.condition( {v:x for v,x in zip(vs,xs)} );

  #def cfg2str(cfg):
  #  return ''.join(str(cfg[i]) if i in cfg else '-' for i in self.X);
     
  def eliminate(self, elimVars, elimOp):
    """Eliminate (remove) a set of variables from the model

    Args:
      elimVars (iterable): list of variables to eliminate (in order of elimination)
      elimOp (str or function): function to eliminate variable v from factor F; 'max', 'min', 'sum', 'lse',
      or a user-defined custom function, called as "elimOp(F,v)" and returning a new Factor.
    """
    if isinstance(elimVars,Var)|isinstance(elimVars,int): elimVars = [elimVars]   # check for single-var case
    if type(elimOp) is str:    # Basic elimination operators can be specified by a string
        elimOp = elimOp.lower()
        if   elimOp == "sum": elimOp = lambda F,X: F.sum(X)
        elif elimOp == "lse": elimOp = lambda F,X: F.lse(X)
        elif elimOp == "max": elimOp = lambda F,X: F.max(X)
        elif elimOp == "min": elimOp = lambda F,X: F.min(X)
        else: raise ValueError("Unrecognized elimination type {}; 'sum','lse','max','min' or custom function".format(elimOp));
    for v in elimVars:
      flist = self.factorsWith(v)
      if len(flist):
        F = flist[0].copy()
        for f in self.factorsWith(v)[1:]: 
          if self.isLog: F += f
          else:          F *= f
        self.removeFactors(flist)
        F = elimOp(F, [v])
        try:
          tmp = F.vars                # raise exception if no variables (not Factor or similar)
          self.addFactors([F],False)  # add factor F by reference
        except:  
          self.addFactors([Factor([],F)],False)          # scalar => add as scalar factor
          # TODO: could check for existing scalar? or leave as is to preserve independent problem correspondence?
      self.X[v] = Var(int(v),1)    # remove "concept" of variable v ( => single state)

  def joint(self):
    """Compute brute-force joint function F(x) = \\prod_r f_r(x_r) as a (large) factor"""
    F = self.factors[0].copy()
    for f in self.factors[1:]: 
      if self.isLog: F += f
      else:          F *= f   
    return F

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
        
    # Note: to split graph into connected components, use e.g.
    # >>> [GraphModel(model.factorsWithAny(vs)) for vs in model.connectedComponents()]
    # (although each model may have the full set of variables, most with 0 or 1 state...)  
    # NOTE: don't forget about any scalar factors: [f for f in model.factors where f.nvar==0]
    
    
  ############# FUNCTIONS TODO ######################
  
  # Algorithms:
  #   (CSP) AC, Backtrack, Local Search;  (GEN) gibbs, mh?, MAS, MBE, WMB, local search, other search?
  # Train:
  #   CD, ???
 
  # "MRF" object (variable dependencies only)?  





#### Methods needed
# display / __repr__

# junction tree / wmb?

# sample

# gibbs, MH (?), ... trainCD?


#class MRF:
#
#

def bnOrder(bn):
    """Return a topological order for the variables (roots to leaves), assuming 'bn' is a Bayes net

    Returns:
        list: variable IDs in a topological order from roots to leaves
    """
    temp = GraphModel(bn.factors, copy=False)  # slow? unnecessary?
    nvar = len(bn.vars_nontrivial)
    topo_order = [-1]*nvar
    for i in range(nvar):
        #print(f"Have: {[f.vars for f in temp.factors]}")
        # check if all remaining vars have only one state?
        if temp.factors[0].nvar != 1: return None    # if there are no root variables, not a BN
        X = temp.factors[0].vars[0]                  # otherwise, add root to the topological order
        if X.states < 2: raise ValueError('Factor with trivial variable argument?')
        topo_order[i] = X.label
        #print(f"  Root {X.label}")
        withX = temp.factorsWith(topo_order[i])      # remove it from its childrens' CPTs
        temp.removeFactors(withX)                    #  to look for the next variable in the order
        temp.addFactors( [f.condition2([X],[0]) for f in withX if f.nvar > 1] )
        #print(f"  Now {[f.vars for f in temp.factors]}")
    return topo_order


def bnPrune(bn, query):
  """Prune all non-ancestors of 'query' in Bayesian network 'bn'
  """
  # TODO: implement: find all ancestors of query; remove non-ancestors (do bn-check elim for safety?)
  raise NotImplementedError('Not implemented')


def bnSample(model, order, evidence={}):
  """xs, lnP = bnSample(model, order, evidence)
  Draw a sample from a Bayes net with given topological order and evidence E={ Xi: k ... }

  Args:
    model (GraphModel): A Bayesian network or other graphical model
    order (list): A topological ordering of the variables, e.g., from ``bnOrder(model)``

  Returns:
     tuple : xs, the sampled configuration (including evidence)
     float : lnP, the log-probability of the sampled configuration xs 

  Notes: lnP, the log-probability, does not include any evidence probabilities.  If `model` is not a
     Bayes net, the process and lnP will still correspond to some valid distribution over X given E.
  """
  lnP, cfg = 0.0, evidence.copy()
  pri = np.zeros((max(order)+1,))-1
  pri[order] = np.arange(len(order))
  for i in order:
    if i in evidence: continue
    F = model.factorsWith(i)
    fid = np.argmin( [ np.max(pri[f.vars.labels]) for f in F ] )  # find factor with X and earliest neighbors
    Pi = F[fid].condition(cfg)
    if model.isLog: Pi.expIP()    # if storing log-factors, exponentiate...
    Pi /= Pi.sum()
    Pi = Pi.marginal([i])     # likely not necessary, but to be safe for non BNs
    cfg[i] = Pi.sample()[0]
    lnP += np.log( Pi[cfg[i]] )
  return tuple(cfg.get(i,0) for i in model.X), lnP
  # TODO: if model.X is not 0...n-1, this does not include missing variables (?)

# One-line likelihood weighting estimator:
# np.mean( [ np.exp(model.logValue(xs)-lnP) for s in range(1000) for xs,lnP in [gm.bnSample(model,order,evid)] ] )
#
#



#def _eliminationOrder_OLD(gm, orderMethod=None, nExtra=-1, cutoff=inf, priority=None, target=None):
#  """Find an elimination order for a graphical model
#  Args:
#    gm (GraphModel): A graphical model object
#    method (str): Heuristic method; one of {'minfill','wtminfill','minwidth','wtminwidth','random'}
#    nExtra (int): Randomly select eliminated variable from among the best plus nExtra; this adds
#        randomness to the order selection process.  0 => randomly from best; -1 => no randomness (default)
#    cutoff (float): Quit early if ``score`` exceeds a user-supplied cutoff value (returning ``target, cutoff``)
#    target (list): If the identified order is better than cutoff, write it directly into passed ``target`` list
#    priority (list, optional): Optional list of variable priorities; lowest priority variables are 
#        eliminated first.  Useful for mixed elimination models, such as marginal MAP inference tasks.
#  Returns:
#    list: The identified elimination order
#    float: The "score" of this ordering
#  Using ``target`` and ``cutoff`` one can easily search for better orderings by repeated calls:
#  >>> ord, score = eliminationOrder(model, 'minfill', nExtra=2, cutoff=score, target=ord) 
#  """
#  import bisect
#  orderMethod = 'minfill' if orderMethod is None else orderMethod.lower()
#  priority = [1 for x in gm.X] if priority is None else priority
#
#  if   orderMethod == 'minfill':    score = lambda adj,Xj: sum([0.5*len(adj[Xj]-adj[Xk]) for Xk in adj[Xj]])
#  elif orderMethod == 'wtminfill':  score = lambda adj,Xj: sum([(adj[Xj]-adj[Xk]).nrStatesDouble() for Xk in adj[Xj]])
#  elif orderMethod == 'minwidth':   score = lambda adj,Xj: len(adj[Xj])
#  elif orderMethod == 'wtminwidth': score = lambda adj,Xj: adj[Xj].nrStatesDouble()
#  elif orderMethod == 'random':     score = lambda adj,Xj: np.random.rand()
#  else: raise ValueError('Unknown ordering method: {}'.format(orderMethod))
#
#  adj = [ VarSet([Xi]) for Xi in gm.X ]
#  for Xi in gm.X: 
#    for f in gm.factorsWith(Xi, copy=False):
#      adj[Xi] |= f.vars
#
#  # initialize priority queue of scores using e.g. heapq or sort
#  scores  = [ (priority[Xi],score(adj,Xi),Xi) for Xi in gm.X ]
#  reverse = scores[:]
#  scores.sort()
#  totalSize = 0.0
#  _order = [0 for Xi in gm.X]
#
#  for idx in range(gm.nvar):
#    pick = 0
#    Pi,Si,Xi = scores[pick]
#    if nExtra >= 0:
#      mx = bisect.bisect_right(scores, (Pi,Si,gm.X[-1]))  # get one past last equal-priority & score vars
#      pick = min(mx+nExtra, len(scores))                  # then pick a random "near-best" variable
#      pick = np.random.randint(pick)
#      Pi,Si,Xi = scores[pick]
#    del scores[pick]
#    _order[idx] = Xi.label        # write into order[idx] = Xi
#    totalSize += adj[Xi].nrStatesDouble()
#    if totalSize > cutoff: return target,cutoff  # if worse than cutoff, quit with no changes to "target"
#    fix = VarSet()
#    for Xj in adj[Xi]:
#      adj[Xj] |= adj[Xi]
#      adj[Xj] -= [Xi]
#      fix |= adj[Xj]   # shouldn't need to fix as much for min-width?
#    for Xj in fix:
#      Pj,Sj,Xj = reverse[Xj]
#      jPos = bisect.bisect_left(scores, (Pj,Sj,Xj))
#      del scores[jPos]                        # erase (Pj,Sj,Xj) from heap 
#      reverse[Xj] = (Pj,score(adj,Xj),Xj)     
#      bisect.insort_left(scores, reverse[Xj]) # add (Pj,score(adj,Xj),Xj) to heap & update reverse lookup
#  if not (target is None): 
#    target.extend([None for i in range(len(target),len(_order))])  # make sure order is the right size
#    for idx in range(gm.nvar): target[idx]=_order[idx]   # copy result if completed without quitting
#  return _order,totalSize


def eliminationOrder(gm, orderMethod=None, nExtra=-1, cutoff=inf, priority=None, target=None):
  """Find an elimination order for a graphical model

  Args:
    gm (GraphModel): A graphical model object
    method (str): Heuristic method; one of {'minfill','wtminfill','minwidth','wtminwidth','random'}
    nExtra (int): Randomly select eliminated variable from among the best plus nExtra; this adds
        randomness to the order selection process.  0 => randomly from best; -1 => no randomness (default)
    cutoff (float): Quit early if ``score`` exceeds a user-supplied cutoff value (returning ``target, cutoff``)
    priority (list, optional): Optional list of variable priorities; lowest priority variables are 
        eliminated first.  Useful for mixed elimination models, such as marginal MAP inference tasks.
    target (list): If the identified order is better than cutoff, write it directly into passed ``target`` list

  Returns:
    list: The identified elimination order
    float: The "score" of this ordering

  Using ``target`` and ``cutoff`` one can easily search for better orderings by repeated calls:
  >>> ord, score = eliminationOrder(model, 'minfill', nExtra=2, cutoff=score, target=ord) 
  """
  import bisect
  orderMethod = 'minfill' if orderMethod is None else orderMethod.lower()
  priority = [1 for x in gm.X] if priority is None else priority

  if   orderMethod == 'minfill':    score = lambda adj,Xj: 0.5*sum([len(adj[Xj]-adj[Xk]) for Xk in adj[Xj]])
  elif orderMethod == 'wtminfill1': score = lambda adj,Xj: sum([Xk.states*Xl.states for Xk in adj[Xj] for Xl in adj[Xj]-adj[Xk]])
  elif orderMethod == 'wtminfill':  score = lambda adj,Xj: sum([(adj[Xj]-adj[Xk]).nrStatesDouble() for Xk in adj[Xj]])
  elif orderMethod == 'minwidth':   score = lambda adj,Xj: len(adj[Xj])
  elif orderMethod == 'wtminwidth1': score = lambda adj,Xj: sum([Xk.states*Xl.states for Xk in adj[Xj] for Xl in adj[Xj]-[Xk]])
  elif orderMethod == 'wtminwidth': score = lambda adj,Xj: adj[Xj].nrStatesDouble()
  elif orderMethod == 'random':     score = lambda adj,Xj: np.random.rand()
  else: raise ValueError('Unknown ordering method: {}'.format(orderMethod))

  adj = [ gm.markovBlanket(Xi) for Xi in gm.X ]  # build MRF

  # initialize priority queue of scores using e.g. heapq or sort
  reverse  = [ (priority[Xi],score(adj,Xi),Xi) for Xi in gm.X ]
  scores = SortedSet( reverse ); 
  totalSize = 0.0
  #_order = np.zeros((len(gm.X),)) #np.array([0 for Xi in gm.X])
  _order = [0]*len(gm.X)

  for idx in range(gm.nvar):
    pick = 0
    Pi,Si,Xi = scores[pick]
    if nExtra >= 0:
      mx = bisect.bisect_right(scores, (Pi,Si,gm.X[-1]))  # get one past last equal-priority & score vars
      pick = min(mx+nExtra, len(scores))                  # then pick a random "near-best" variable
      pick = np.random.randint(pick)
      Pi,Si,Xi = scores[pick]
    del scores[pick]
    _order[idx] = Xi.label        # write into order[idx] = Xi
    totalSize += adj[Xi].nrStatesDouble()
    if totalSize > cutoff: return target,cutoff  # if worse than cutoff, quit with no changes to "target"
    fix = adj[Xi].copy()          # fix up (at least) scores of variables adjacent to Xi
    for Xj in adj[Xi]:            # update Markov graph to reflect removal of Xi: join Xi's neighbors
      adj[Xj] |= adj[Xi]
      adj[Xj] -= [Xi,Xj]  # TODO adj[Xj].remove(Xi) slightly faster but still unsupported by cython version
      fix |= adj[Xj]              # also fix scores of 2nd neighbors for min-fill (shouldn't need to fix as much for min-width?)
    for Xj in fix:                # Now, update the scores of "fix" set in the priority queue:
      Pj,Sj,Xj = reverse[Xj]
      scores.remove(reverse[Xj])
      reverse[Xj] = (Pj,score(adj,Xj),Xj)  # re-score Xj
      scores.add(reverse[Xj])     # add (Pj,score(adj,Xj),Xj) to heap & update reverse lookup
  if not (target is None): 
    target.extend([None for i in range(len(target),len(_order))])  # make sure order is the right size
    for idx in range(gm.nvar): target[idx]=_order[idx]   # copy result if completed without quitting
  return _order,totalSize
    

class PseudoTree(object):
  """Represent the pseudo-tree of a graphical model, given elimination ordering

  Attributes:
     pt.parent (list): pt.parent[x] gives the earliest parent of x in the pseudotree
     pt.width (int): width (largest clique) in the tree
     pt.depth (int): depth (longest chain of conditionally dependent variables) in the tree; = n for or-chain
     pt.size  (float): total # of operations (sum of clique sizes) for the elimination process
  """
  def __init__(self, model, elimOrder, force_or=False, max_width=None):
    """Build the pseudotree. Set force_or=True to force an or-chain pseudotree."""
    self.order  = elimOrder;
    self.parent = [None]*len(elimOrder);
    self.width = 0;
    self.depth = 0;
    self.size  = 0.0;
    height = np.zeros((len(elimOrder),),dtype=int);
    priority = np.zeros((len(elimOrder),),dtype=int);
    priority[elimOrder] = np.arange(len(elimOrder));
    if max_width is None: max_width = len(elimOrder);

    adj = [ model.markovBlanket(Xi) for Xi in model.X ]  # build MRF
 
    for i,x in enumerate(elimOrder):
      nbrs = adj[x].copy();      # when we eliminate x,
      for y in nbrs:      #   we connect all its neighbors to each other
        adj[y] |= nbrs;
        adj[y] -= [x,y];
      self.width  = max(self.width, len(nbrs));   # update width statistic
      if self.width > max_width: self.depth,self.width = inf,inf; return  # give up?
      self.size  += nbrs.nrStatesDouble()         #   and total size statistic
      if not force_or:    # and-or tree: find earliest eliminated neighbor 
        if len(nbrs): self.parent[x] = elimOrder[ min(priority[nbrs.labels]) ] 
        #if self.parent[x] is not None: assert( priority[self.parent[x]] > priority[x] )
      else:               # force or-tree (chain) pseudotree
        if i != len(elimOrder)-1: self.parent[x] = elimOrder[i+1];
      if self.parent[x] is not None:
        height[self.parent[x]] = max(height[self.parent[x]], height[x]+1);
      #ALT: if self.parent[x] is not None and height[x] >= height[self.parent[x]]: height[self.parent[x]]=height[x]+1

    self.depth = max(height);

  def orderDFS(self, priority=None):
    """Find an elimination order corresponding to a (reverse) depth-first traversal of the pseudotree"""
    children = [ [] for v in self.order ]
    # First find a DFS traversal of the pseudo-tree
    for i,p in enumerate(self.parent): children[p] += [i]
    queue = [ i for i,p in self.parent if p is None ]
    while len(queue):
      new_order += queue.pop()            # get next variable off queue
      queue += children[ new_order[-1] ].reverse()  # append children to queue
    assert( len(new_order) == len(self.order) )
    # Now, re-order "new_order" to respect the priority structure if necessary:
    if priority is not None:
      new_order = [(priority[i],o,i) for o,i in enumerate(new_order) ]
      new_order.sort()
      new_order = [i for p,o,i in new_order]  # re-extract variable indices
    return new_order 


### TODO
# Useful ordering operations?
#  Reorder to minimize memory if cleared sequentially?  Calc such requirement?
#  Reorder to root jtree at clique?
#  Reorder to minimize depth?
#  




# TODO: function to return bayes ordering of factors from the model given topo ordering of X?
#   var -> priority; place each factor by last priority; 
#       if conflict, score entropy H(xi|xpa) to decide
#    
def factorOrder(factors, varOrder):
  """Return an order of factors for sampling given a variable order for sampling"""
  pri = [0 for x in varOrder]
  for i,x in enumerate(varOrder):          # first, find position of each var in sampling order
    pri[x]=i
  factorOrder = [ Factor() for x in varOrder ]  # fill order with blanks initially
  for f in factors:
    f_pri = max([pri[x] for x in f.vars])  # get last-sampled variable for this factor
    if factorOrder[f_pri].nvar == 0:
      factorOrder[f_pri] = f               # if first factor for this variable, save it
    else:                                  # o.w. take one with the lowest conditional entropy:
      if ent[f_pri] < 0:                   #   (compute previous' if necessary)
        ent[f_pri] = factorOrder[f_pri].entropy() - factorOrder[f_pri].sum([f_pri]).entropy()
      ent_new = f.entropy() - f.sum([f_pri]).entropy()  # (and this factor's)
      if ent_new < ent[f_pri]:             #   (keep whichever is lower)
        factorOrder[f_pri] = f
        ent[f_pri] = ent_new
  return factorOrder
  

################################################################################################
# Basic sampling procedures: 
# Should sample from p(x) if it is a Bayes net with known topological ordering
# Sample from some simple proposal based on the factors of p(x) if not.

#
# TODO: forward sample: draw each x; track normalization constant (in case non-norm'd) & scalar f'ns; return x,w
#   => rejection sampling?  importance sampling? others?
#   extra param stores / filled with factors in order if not None?

# sampling object?
#   constructor takes factors, variable order if avail, and 
#    constructs ordered seq. of conditionals (by ref if possible) by selecting the largest
#    factor containing only preceding vars & conditioning if reqiured.
#    => BN + order = factor sequence for sampling
#    => BP beliefs + order = BP BN proposal
# ?? do without actually conditioning? condition on the fly: 
#   compute the norm, sample, etc. manually in f'n

#def sampleSequential(model, varOrder, factorOrder=None):
#  """Draw a configuration X sequentially from the model factors.
#  For a Bayes net using a valid topological order, this is equivalent to forward sampling.
# 
#  Returns xs,lnQ : the sampled config xs (tuple) and its log-probability under the sampling distribution (float)
#  """
#  if factorOrder is None: factorOrder = []
#  if len(factorOrder)==0:
#    raise NotImplementedError
#    # figure out an ordering of the factors in "model" to use & store them in factorOrder:
#    # for each xi in varOrder:
#    #   from model, get factors with xi
#    #   from small->large (?) check if args all in "done"; if so, add to factorOrder
#    #   add xi to done
#  s = {}                              # storage for sample value
#  lnP = 0.0                           # log p(x) of drawn sample
#  # TODO: enumerate over varOrder instead; use factorOrder[i], then weight by factorOrd[i:] after
#  for i,f in enumerate(factorOrder):  # sample Xi from F[i | prev]:
#    xi  = varOrder[i]
#    Pxi = f.condition( s )
#    # TODO: check Pxi over one var, xi
#    Zi  = Pxi.sum()
#    s[ xi ], = Pxi.sample(Z=Zi)  # draw sample for i'th variable
#    lnP += np.log( Pxi[s[xi]] / Zi )
#  s_list = [ s[i] for i in range(len(s)) ]
#  return s_list, lnP
   





