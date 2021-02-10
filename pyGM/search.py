# Search-based algorithms
#

import numpy as np
import time as time
import heapq
from .factor import *
from .wmb import *
from .wogm import *

"""
Search algos:
  local greedy search; local greedy + stochastic search; local stochastic search
  depth first B&B
  A*
  RBFS?
  ...
"""


class NodeDFS(object):
  def __init__(self,parent,X=None,x=None,val=-inf):
    self.parent = parent
    self.children = []
    #self.priority = -inf
    self.value = val
    self.heuristic = None
    self.X = X
    self.x = x   # node is for assignment X=x given ancestor assigments



class PrunedDFS(object):
  """Depth-first search using pruning."""
  def __init__(self, model, heuristic, weights='max', xinit={}):
    self.model = model
    self.heuristic = heuristic
    if len(xinit): self.xhat = xinit; self.L = model.logValue(xinit) # use initial config
    else:          self.xhat = {};    self.L = -inf
    #if   weights == 'max': self.weights = [0.0 for X in model.X]
    #elif weights == 'sum': self.weights = [1.0 for X in model.X]
    #else:                  self.weights = weights
    self.root = NodeDFS(None);
    self.node = self.root
    self.root.heuristic = heuristic
    self.cfg = {}

  def done(self):
    return self.node is None

  # def upperbound(self): = max(self.L,max([n.value for n in self.node to root]))
  # def lowerbound(self): = self.L

  def __next(self,n):
    """Helper to return next node in DFS queue"""
    while n is not None:
      if n.children and not self.prune(n.children[-1]):
        n = n.children.pop()            # if we can move downward, do so;
        break
      else:                             # otherwise, move to parent/sibling
        self.cfg.pop(n.X,None) 
        n = n.parent
    if n is not None: self.cfg[n.X] = n.x;
    return n, self.cfg
        
  def isLeaf(self,n): 
    return len(self.cfg) == self.model.nvar  # config for all variables => done (or-tree)

  def prune(self,n):
    return n.value <= self.L            # value (upper bound) no better than current soln

  def run(self, stopNodes=inf, stopTime=inf, verbose=False):
    nnodes = 0; stopTime += time.time();           # initialize stopping criteria
    while not self.done() and nnodes<stopNodes and time.time()<stopTime:
        nnodes += 1; n = self.node;

        if self.isLeaf(n):              # for leaf nodes, can evaluate score directly
          n.value = self.model.logValue(self.cfg); 
          if n.value > self.L:          #    MPE-specific! TODO
            self.L = n.value; 
            self.xhat = self.cfg.copy(); 
            if verbose: print("[{}]".format(self.L),[self.xhat[v] for v in self.model.X])
        else:                          # else, update heuristic given n.X = n.x
          n.heuristic, n.value = n.heuristic.update(self.cfg,n.X,n.x)

        if not self.prune(n):           # if not pruned, expand n & generate children:
          X,vals = n.heuristic.next_var(self.cfg,n.X,n.x)
          idx = np.argsort(vals)
          n.children = [NodeDFS(n,X,i,vals[i]) for i in idx]
          for c in n.children: c.heuristic = n.heuristic   # point to previous heuristic

        # Now move on to the next node in the queue
        self.node, self.cfg = self.__next(n) 
        


class NodeAStar(object):
  def __init__(self,parent,X=None,x=None,val=-inf):
    self.parent = parent
    self.children = []
    #self.priority = -inf
    self.value = val
    self.heuristic = None
    self.X = X
    self.x = x   # node is for assignment X=x given ancestor assigments

class AStar(object):
  """A-Star heuristic search """
  def __init__(self, model, heuristic, weights='max', xinit={}):
    self.model = model
    self.heuristic = heuristic
    if len(xinit): self.xhat = xinit; self.L = model.logValue(xinit) # use initial config
    else:          self.xhat = {};    self.L = -inf
    #if   weights == 'max': self.weights = [0.0 for X in model.X]
    #elif weights == 'sum': self.weights = [1.0 for X in model.X]
    #else:                  self.weights = weights
    self.root = NodeAStar(None);
    self.node = self.root
    self.root.heuristic = heuristic
    self.cfg = {}                    # storage for partial configuration of self.node

  def done(self,):
    return self.node is None

  # def upperbound(self): = self.root.value
  # def lowerbound(self): = self.L ( = -inf if not done )

  def __next(self,n):
    """Helper to return next node in priority queue"""
    # First, travel from n to the root, updating the parent's priority & the children's ordering
    while n.parent is not None: 
      heapq.heapreplace(n.parent.children, (-n.value, n))
      n = n.parent
      ## TODO: update both "value" and "priority" based on "weight" of n.X
      if n.value == n.children[0][1].value: break;  # didn't change value => stop propagating
      n.value = n.children[0][1].value
    
    n = self.root                           # now find highest priority leaf:
    cfg = {}
    while n.children: 
      n = n.children[0][1]  # follow top priority child downward
      cfg[n.X] = n.x        # and track the associated configuration
    return n, cfg
         
  def isLeaf(self,n):
    return len(self.cfg) == self.model.nvar  # config for all variables => done (or-tree)

  def run(self, stopNodes=inf, stopTime=inf, verbose=False):
    nnodes = 0; stopTime += time.time();           # initialize stopping criteria
    while not self.done() and nnodes<stopNodes and time.time()<stopTime:
        nnodes += 1; n = self.node;

        if self.isLeaf(n):              # for leaf nodes, can evaluate score directly
          n.value = self.model.logValue(self.cfg);
          self.xhat = self.cfg.copy()   #    MPE-specific! TODO
          if verbose: print("[{}]".format(n.value),[self.xhat[v] for v in self.model.X])
          self.node, self.cfg = self.__next(n)  # go to next node; if it's this node, we're done!
          if self.node == n: self.node = None; break;     # TODO: hacky
        else:                          # else, update heuristic given n.X = n.x
          n.heuristic, n.value = n.heuristic.update(self.cfg,n.X,n.x)
          X,vals = n.heuristic.next_var(self.cfg,n.X,n.x)
          idx = np.argsort(vals)
          n.children = [(-vals[i],NodeAStar(n,X,i,vals[i])) for i in idx]
          heapq.heapify(n.children)
          for c in n.children: c[1].heuristic = n.heuristic   # point to previous heuristic

        # Now move on to the next node in the queue
        self.node, self.cfg = self.__next(n)
        #print(" => ",self.cfg)



"""
Search Heuristics should have two functions:
  new_heuristic, node_value = Heuristic.update( config, Var, val )
    config: map or list with current partial configuration
    Var,val: most recent assignment Var=val 
    new_heuristic : copy of new heuristic if dynamic; pointer to old if not
    node_value : updated heuristic value of configuration

  Var,scores = Heuristic.next_var(config,Var,val)
    Return next variable to expand & preliminary heuristic of its values for ordering
"""


class WmbStatic(object):
  """Static heuristic function based on a (weighted) minibucket bound"""
  def __init__(self, model, *args, **kwargs):
    self.wmb = WMB(model,*args,**kwargs);
    self.bound = self.wmb.msgForward(1.0,0.1)
    self.wmb.initHeuristic()

  def update(self, config, Xi, xi):
    """Condition the heuristic on new assignment Xi=xi (no effect for static heuristic)"""
    if Xi is None: return self, self.bound
    return self, self.wmb.resolved(Xi,config)+self.wmb.heuristic(Xi,config)

  def next_var(self, config, Xi, xi):
    """Select next unassigned variable & preliminary scores for ordering"""
    p = self.wmb.priority[Xi] if Xi is not None else self.wmb.model.nvar
    X = self.wmb.X[self.wmb.elimOrder[p-1]] if p else None
    vals = []
    for x in range(X.states):   # TODO: SLOW
      config[X] = x
      vals.append( self.wmb.resolved(X,config)+self.wmb.heuristic(X,config) )
    config.pop(X)
    return X, vals


class WmbDynamic(object):
  """Dynamic heuristic function based on a (weighted) minibucket bound"""
  def __init__(self, model, *args, **kwargs):
    self.wogm = WOGraphModel(model.factors,*args,isLog=model.isLog,**kwargs)
    self.wogm.init()
    self.wogm.update(stopIter=2,stopTime=1.0)
    self.bound = self.wogm.factors[0][0]
    #self.wmb = WMB(model,*args,**kwargs);
    #self.bound = self.wmb.msgForward(1.0,0.1)
    #self.wmb.initHeuristic()

  def update(self, config, Xi, xi):
    """Condition the heuristic on new assignment Xi=xi"""
    if Xi is None: return self, self.bound 
    else: 
      H = WmbDynamic( self.wogm, elimOrder = self.wogm.elimOrder, weights=self.wogm.weights )
      H.wogm.condition({Xi:xi})   # TODO: make more streamlined
      H.wogm.update(stopIter=2,stopTime=1.0)
      return H, H.wogm.factors[0][0]
      #model = self.wmb.model.copy()
      #model.condition({Xi:xi})
      #H = WmbDynamic(model,self.wmb.elimOrder,iBound=1,weights=self.wmb.weights)
      #return H,H.bound

  def next_var(self, config, Xi, xi):
    """Select next unassigned variable & preliminary scores for ordering"""
    X = None
    scores = [ inf if X in config else self.wogm.v_beliefs[X].entropy() for X in self.wogm.X ]
    #scores = [ inf if X in config else -len(self.wogm.factorsWith(X)) for X in self.wogm.X ]
    X = np.argmin(scores)
    vals = self.wogm.costtogo(X).table
    return X, vals
    #for xi in self.wmb.X: 
    #  if xi not in config: X=xi
    #vals = [ self.bound for i in range(X.states) ] if X is not None else []
    #return X,vals



class SimpleStatic(object):
  """Trivial heuristic function for MAP or CSP search"""
  def __init__(self,model,*args,**kwargs):
    self.model = model if model.isLog else model.copy().toLog()

  def update(self,config,Xi,xi):
    bound = sum(( f.condition(config).max() for f in self.model.factors ))
    return self,bound

  def next_var(self,config,Xi,xi):
    X = max( [(len(self.model.factorsWith(X,copy=False)),X) for X in self.model.X if X not in config] )[1]
    val = sum(( f.condition(config).maxmarginal([X]) for f in self.model.factors ))
    return X, val.table





