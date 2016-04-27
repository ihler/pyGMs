"""
graphmodel.py

Defines a graphical model container class for reasoning about graphical models

Version 0.0.1 (2015-09-28)
(c) 2015 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import operator as operator
import numpy as np
from sortedcontainers import SortedSet;

from .factor import *

# Make a simple sorted set of factors, ordered by clique size, then lexicographical by scope
def factorSet(it=None): return SortedSet(iterable=it,key=lambda f:'{}.'.format(f.nvar)+str(f.vars)[1:-1],load=30)
# So, fs = factorSet([list]) will make sure:
#   fs[0] is a minimal factor (smallest # of variables) and fs[-1] is a maximal factor (largest # of variables)

class GraphModel(object):
  """A graphical model class"""

  X            = []            # variables model defined over
  factors      = factorSet()   # collection of factors that make up the model, sorted by scope size, lex
  factorsByVar = []            # factor lookup by variable:  var -> factor list 
 
  #TODO: useful stuff for algorithm objects interacting with the graph
  lock         = False         # are factors "locked", or are reparameterization changes OK?
  sig          = 0;            # "signature" of factor cliques; changes if model structure (cliques) changes
  

  def __repr__(self):
    # TODO: finish
    return "Graphical model: {} vars, {} factors".format(self.nvar,self.nfactors)

  def __init__(self, factorList=None):
    """Create a graphical model object from a factor list.  Maintains local copies of the factors."""
    self.X = []
    self.factors = factorSet()
    self.factorsByVar = []
    if not (factorList is None): self.addFactors(factorList)
    self.lock = False
    self.sig = 0
    

  def addFactors(self,flist,copy=True):
    """Add a list of factors to the model; factors are copied locally unless copy=False explicitly used"""
    import copy
    if (copy): flist = copy.deepcopy(flist)  # create new factor copies that this model "owns"
    self.factors.update(flist);   # add factor to set of all factors, and to by-variable indexing
    for f in flist:
      # TODO: if constant do something else?  (Currently just leave it in factors list)
      for v in reversed(f.vars):
        if (v.label >= len(self.X)): 
          self.X.extend( [Var(i,0) for i in range(len(self.X),v.label+1) ] )
          self.factorsByVar.extend( [ factorSet() for i in range(len(self.factorsByVar),v.label+1) ] )
        if self.X[v].states == 0: self.X[v] = v   # copy variable info if undefined, then check:
        if self.X[v].states != v.states: raise ValueError('Incorrect # of states',v,self.X[v])
        self.factorsByVar[v].add(f)

  def removeFactors(self,flist):
    """Remove factors from the model; e.g. removeFactors(factorsWith([0])) removes all factors involving x0"""
    self.factors.difference_update(flist)
    for f in flist:
      for v in f.vars:
        self.factorsByVar[v].discard(f)

  def makeCanonical(self):
    """Add/merge factors to make a canonical factor graph: singleton factors plus maximal cliques"""
    for f in self.factors:
      fs = self.factorsWithAll(f.vars)
      if (f.nvar > 1) & (fs[-1] != f):
        fs[-1] *= f
        self.removeFactors([f])
        
  def makeMinimal(self):
    """Merge factors to make a minimal factor graph: retain only factors over maximal cliques"""
    for f in self.factors:
      fs = self.factorsWithAll(f.vars)
      if (fs[-1] != f):
        fs[-1] *= f
        self.removeFactors([f])


  def factorsWith(self,v):
    """gm.factorsWith(v) : get list of factors that include variable v"""
    return self.factorsByVar[v].copy()

  def factorsWithAny(self,vs):
    """gm.factorsWithAny([vs]) : get list of factors that include any variables in list vs"""
    flist = factorSet()
    for v in vs: flist.update( self.factorsByVar[v] )
    return flist

  def factorsWithAll(self,vs):
    """gm.factorsWithAll([vs]) : get list of factors that include all variables in list vs"""
    if (len(vs)==0): return self.factors.copy()
    flist = self.factorsByVar[vs[0]].copy()
    for v in vs: flist.intersection_update(self.factorsByVar[v])
    return flist
  
  def markovBlanket(self,v):
    """Get the Markov blanket of variable v (all vars involved in a factor with v)"""
    vs = VarSet()
    for f in self.factorsByVar[v]: vs |= f.vars
    vs -= [v]
    return vs
 
  def value(self,x):
    """Evaluate F(x) = \prod_r f_r(x_r) for some (full) configuration x"""
    return np.product( [ f.valueMap(x) for f in self.factors ] )

  def logValue(self,x): 
    """Evaluate log F(x) = \sum_r log f_r(x_r) for some (full) configuration x"""
    return sum( [ np.log(f.valueMap(x)) for f in self.factors ] )


  def isBinary(self):   # Check whether the model is binary (all variables binary)
    """Check whether the graphical model is binary (all variables <= 2 states)"""
    return all( [x.states <= 2 for x in self.X] )

  def isPairwise(self): # Check whether the model is pairwise (all factors pairwise)
    """Check whether the graphical model is pairwise (max scope size = 2)"""
    return all( [len(f.vars)<= 2 for f in self.factors] )

  def isCSP(self): 
    """Check whether the graphical model is a valid CSP (all zeros or ones)"""
    isTableCSP = lambda t : all( [ ((v==0.0) | (v==1.0)) for v in np.nditer(t) ] )
    return all( [isTableCSP(f.table) for f in self.factors] )

  #def isBN(self, order):  # Check whether the model is a valid Bayes Net with topo order "order"
  # TODO
  # convert order to priority listing; found = [0 for x in X]
  # run through factors, computing highest priority in each
  #   find xlast = x : pri(x) == max(pri(xalpha))
  #   check that f.sum(xlast) == 1 ; if not fail, if so set found(xlast)
  # set found(x) = 1 for x : x.states < 2
  # check that found == 1

  @property
  def vars(self):
    """List of variables in the graphical model"""
    return self.X

  def var(self,i):   # TODO: change to property to access (read only?) X?
    """Return a variable object (with # states) for id 'i'"""
    return self.X[i]

  @property
  def nvar(self):
    """The number of variables ( = largest variable id) in the model"""
    return len(self.X)

  @property
  def nfactors(self): 
    """The number of factors in the model"""
    return len(self.factors)



  def condition2(self, vs, xs):
    """Condition / clamp the graphical model on the partial configuration vs=xs (may be lists)"""
    constant = 0.0
    for f in self.factorsWithAny(vs): 
      self.removeFactors([f])
      fc = f.condition2(vs,xs)   # TODO: want to keep "f" reference?  f.__build?  (still  remove/add for sort)
      if (fc.nvar == 0): constant += np.log(fc[0])  # if it's now a constant, just pull it out 
      else: self.addFactors([fc])                   # otherwise add the factor back to the model
    # TODO: merge any subsumed factors?  (no; use minimal/canonical for that?)
    # Simpler: track modified factors; get factorsWithAll(f.vars).largest() and merge into that
    constant = np.exp(constant / len(vs))
    for i,v in enumerate(vs):    # add delta f'n factor for each variable
      f = Factor(v,0.0)
      f[xs[i]] = constant        # assignment is nonzero; distribute constant value into these factors
      self.addFactors([f])

  def condition(self, evidence):
    """Condition / clamp the graphical model on a partial configuration (dict) {Xi:xi,Xj:xj...}"""
    return self.condition2([self.X[k] for k,v in evidence.iteritems()], [v for k,v in evidence.iteritems()])
     
  def eliminate(self, elimVars, elimOp):
    """Eliminate (remove) a set of variables; elimOp(F,v) should eliminate variable v from factor F."""
    if isinstance(elimVars,Var)|isinstance(elimVars,int): elimVars = [elimVars]   # check for single-var case
    for v in elimVars:
      F = Factor([],1.0)
      for f in self.factorsWith(v): 
        F *= f
        self.removeFactors([f])
      F = elimOp(F, [v])
      if isinstance(F, Factor): self.addFactors([F],False)  # add factor F by reference
      else:  self.addFactors([Factor([],F)],False)          # scalar => add as scalar factor
      # TODO: could check for existing scalar? or leave as is to preserve independent problem correspondence?
    self.X[v] = Var(int(v),1)    # remove "concept" of variable v ( => single state)

  def joint(self):
    """Compute brute-force joint function F(x) = \prod_r f_r(x_r) as a (large) factor"""
    F = Factor([],1.0)
    for f in self.factors: F *= f
    return F

  # TODO: add function(s) to split graph into any disconnected components?  detect disconnected components?



  def nxMarkovGraph(self):
    """Get networkx Graph object of the Markov graph of the model"""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from( [v.label for v in self.X] )  # TODO: only non-empty / non-trivial variables?
    for f in self.factors:
      for v1 in f.vars:
        for v2 in f.vars:
          if (v1 != v2): G.add_edge(v1.label,v2.label)
    return G 
    """ Plotting examples:
    fig,ax=plt.subplots(1,2)
    pos = nx.spring_layout(G) # so we can use same positions multiple times...
    # use nodelist=[nodes-to-draw] to only show nodes in model
    nx.draw(G,with_labels=True,labels={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6'},
              node_color=[.3,.3,.3,.7,.7,.7,.7],vmin=0.0,vmax=1.0,pos=pos,ax=ax[0])
    nx.draw(G,with_labels=True,labels={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6'},
              node_color=[.3,.3,.3,.7,.7,.7,.7],vmin=0.0,vmax=1.0,pos=pos,ax=ax[1])
    """

  def drawMarkovGraph(self,**kwargs):
    """Draw a Markov random field using networkx function calls"""
    # TODO: fix defaults; specify shape, size etc. consistent with FG version
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from( [v.label for v in self.X if v.states > 1] )  # only non-trivial vars
    for f in self.factors:
      for v1 in f.vars:
        for v2 in f.vars:
          if (v1 != v2): G.add_edge(v1.label,v2.label)
    kwargs['var_labels'] = kwargs.get('var_labels',{n:n for n in G.nodes()})
    kwargs['labels'] = kwargs.get('var_labels',{})
    nx.draw(G,**kwargs)



  def drawFactorGraph(self,var_color='w',factor_color=(.2,.2,.8),**kwargs):
    """Draw a factorgraph using networkx function calls"""
    # TODO: specify var/factor shape,size, etc.
    import networkx as nx
    G = nx.Graph()
    vNodes = [v.label for v in self.X]   # TODO: only non-empty / non-trivial variables?
    fNodes = [-i-1 for i in range(len(self.factors))]
    G.add_nodes_from( vNodes )
    G.add_nodes_from( fNodes )
    for i,f in enumerate(self.factors):
      for v1 in f.vars:
        G.add_edge(v1.label,-i-1)
        #G.add_edge(v1.label,-self.factors.index(f)-1)
    pos = nx.spring_layout(G)   # so we can use same positions multiple times...
    kwargs['var_labels']  =kwargs.get('var_labels',{n:n for n in vNodes})
    #kwargs['var_color']   =kwargs.get('var_color','w')
    #kwargs['factor_color']=kwargs.get('factor_color',(.2,.2,.8))
    kwargs['labels']=kwargs.get('var_labels',{})
    nx.draw_networkx(G,pos, nodelist=vNodes,node_color=var_color,**kwargs)
    kwargs['labels']=kwargs.get('factor_labels',{})
    nx.draw_networkx_nodes(G,pos, nodelist=fNodes,node_color=factor_color,node_shape='s',**kwargs)
    nx.draw_networkx_edges(G,pos,**kwargs)
    #if 'labels' in kwargs:
    #  nx.draw_networkx_labels(G,pos=pos,**kwargs)
    #else:
    #  nx.draw_networkx_labels(G,pos=pos,labels={n:n for n in vNodes},**kwargs)


#  def nxFactorGraph(self):
#    """Get networkx Graph object of the factor graph of the model"""
#    import networkx as nx
#    G = nx.Graph()
#    vNodes = [v.label for v in self.X]
#    fNodes = [-i-1 for i in range(len(self.factors))]
#    G.add_nodes_from( vNodes )
#    G.add_nodes_from( fNodes )
#    for f in self.factors:
#      for v1 in f.vars:
#        G.add_edge(v1.label,-self.factors.index(f)-1)
#    return G 
#    """ Plotting examples:
#    pos = nx.spring_layout(G) # so we can use same positions multiple times...
#    nx.draw_networkx(G,pos=pos, nodelist=[n for n in G.nodes() if n >= 0],node_color='w')
#    nx.draw_networkx_nodes(G,pos=pos, nodelist=[n for n in G.nodes() if n < 0],node_color='g',node_shape='s')
#    nx.draw_networkx_edges(G,pos=pos)
#    nx.draw_networkx_labels(G,pos=pos,labels={n:n for n in G.nodes() if n>=0})
#    """



  ############# FUNCTIONS TODO ######################
  # sampleBN : given a topo ordering, sample in sequence *and* check that each factor is a CPD (?)
  # pseudotree height, width (or vs andor)
  
  # Algorithms:
  #   (CSP) AC, Backtrack, Local Search;  (GEN) gibbs, mh?, MAS, MBE, WMB, local search, other search?
  # Train:
  #   CD, ???
 
  # "MRF" object (variable dependencies only)?  



#  def condition(self, conditionVars, conditionVals ):
#    condFactors = set.union(*[self.adjacency[xi] for xi in conditionVars])
#    for f in condFactors:
#      self.removeFactor(f)
#      self.addFactor( f.condition( conditionVars, conditionVals ) )
#    for v in conditionVars: self.X[v] = Var(v.label,1)   # each xi in cVars has only one possible state now





#### Methods needed
# display / __repr__

# junction tree / wmb?

# sample

# gibbs, MH (?), ... trainCD?

# orderings
# inducedWidth, depth, pseudotree, ...


#class MRF:
#
#

def eliminationOrder(gm, orderMethod=None, nExtra=-1, cutoff=float('inf'), priority=None, target=None):
  '''Find an elimination order for a graphical model
  ord,score = eliminationOrder(gm, method, nExtra, cutoff, priority, target):
    ord: the elimination order (a list of variables)
    score:  the total size of the resulting junction tree (sum of the cliques' sizes)
    gm:     graphical model object
    method: {'minfill','wtminfill','minwidth','wtminwidth','random'}
    nExtra: randomly select eliminated variable from among the best plus nExtra; this adds
            randomness to the order selection process.  0 => randomly from best; -1 => no randomness (default)
    cutoff: quit early if "score" exceeds a user-supplied cutoff value (returning "target,cutoff")
    target:  if identified order is better than cutoff, write directly into passed "target" list
            Note: this means that you can easily search for better orderings by, e.g., repeated calls:
               ord,score = eliminationOrder(gm,'minfill', nExtra=2, cutoff=score, target=ord) 
    priority: optional list of variable priorities; lowest priority variables are eliminated first.
            Useful for mixed elimination models, such as marginal MAP inference tasks
  '''
  import bisect
  orderMethod = 'minfill' if orderMethod is None else orderMethod.lower()
  priority = [1 for x in gm.X] if priority is None else priority

  if   orderMethod == 'minfill':    score = lambda adj,Xj: sum([0.5*len(adj[Xj]-adj[Xk]) for Xk in adj[Xj]])
  elif orderMethod == 'wtminfill':  score = lambda adj,Xj: sum([(adj[Xj]-adj[Xk]).nrStates() for Xk in adj[Xj]])
  elif orderMethod == 'minwidth':   score = lambda adj,Xj: len(adj[Xj])
  elif orderMethod == 'wtminwidth': score = lambda adj,Xj: adj[Xj].nrStates()
  elif orderMethod == 'random':     score = lambda adj,Xj: np.random()
  else: raise ValueError('Unknown ordering method: {}'.format(orderMethod))

  adj = [ VarSet([Xi]) for Xi in gm.X ]
  for Xi in gm.X: 
    for f in gm.factorsWith(Xi):
      adj[Xi] |= f.vars

  # initialize priority queue of scores using e.g. heapq or sort
  scores  = [ (priority[Xi],score(adj,Xi),Xi) for Xi in gm.X ]
  reverse = scores[:]
  scores.sort()
  totalSize = 0.0
  _order = [0 for Xi in gm.X]

  for idx in range(gm.nvar):
    pick = 0
    Pi,Si,Xi = scores[pick]
    if nExtra >= 0:
      mx = bisect.bisect_right(scores, (Pi,Si,gm.X[-1]))  # get one past last equal-priority & score vars
      pick = min(mx+nExtra, len(scores))                  # then pick a random "near-best" variable
      pick = np.random.randint(pick)
      Pi,Si,Xi = scores[pick]
    del scores[pick]
    _order[idx] = Xi        # write into order[idx] = Xi
    totalSize += adj[Xi].nrStates()
    if totalSize > cutoff: return target,cutoff  # if worse than cutoff, quit with no changes to "target"
    fix = VarSet()
    for Xj in adj[Xi]:
      adj[Xj] |= adj[Xi]
      adj[Xj] -= [Xi]
      fix |= adj[Xj]   # shouldn't need to fix as much for min-width?
    for Xj in fix:
      Pj,Sj,Xj = reverse[Xj]
      jPos = bisect.bisect_left(scores, (Pj,Sj,Xj))
      del scores[jPos]                        # erase (Pj,Sj,Xj) from heap 
      reverse[Xj] = (Pj,score(adj,Xj),Xj)     
      bisect.insort_left(scores, reverse[Xj]) # add (Pj,score(adj,Xj),Xj) to heap & update reverse lookup
  if not (target is None): 
    target.extend([None for i in range(len(target),len(_order))])  # make sure order is the right size
    for idx in range(gm.nvar): target[idx]=_order[idx]   # copy result if completed without quitting
  return _order,totalSize
    

### TODO
# Useful ordering operations?
#  Reorder to minimize memory if cleared sequentially?
#  Reorder to root jtree at clique?
#  Reorder to minimize depth?
#  Compute pseudo-tree from graph, order
#  


def bnOrder(factors):
  """Return a topological order for a Bayes net defined by an (ordered) factor list 
     factors[i] should be the conditional probability p(Xi | X_{pa_i}) 
  """
  # TODO: inefficient
  todo = set(fg_alarm.X)
  done = set()
  order = []
  for i in range(len(f_alarm)):
    for j in todo:
      if len(f_alarm[j].vars - done - [fg_alarm.X[j]]) == 0:
        order.append(j.label)
        done.add(j)
        todo.remove(j)
        break
  return order


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

#def forwardSample(model, varOrder, factorOrder=None):
def sampleSequential(model, varOrder, factorOrder=None):
  if factorOrder is None: factorOrder = []
  if len(factorOrder)==0:
    raise NotImplementedError
    # figure out an ordering of the factors in "model" to use & store them in factorOrder:
    # for each xi in varOrder:
    #   from model, get factors with xi
    #   from small->large (?) check if args all in "done"; if so, add to factorOrder
    #   add xi to done
  #x = [-1 for Xi in varOrder]         # storage for sample value
  s = {}                              # storage for sample value
  #lnP, lnW = 0.0, 0.0                 # log p(x) of drawn sample & log w(x) = f(x)/p(x)
  lnP = 0.0                           # log p(x) of drawn sample
  # TODO: enumerate over varOrder instead; use factorOrder[i], then weight by factorOrd[i:] after
  for i,f in enumerate(factorOrder):  # sample Xi from F[i | prev]:
    xi  = varOrder[i]
    Pxi = f.condition( s )
    # TODO: check Pxi over one var, xi
    Zi  = Pxi.sum()
    s[ xi ], = Pxi.sample(Z=Zi)  # draw sample for i'th variable
    lnP += np.log( Pxi[s[xi]] / Zi )
  s_list = [ s[i] for i in range(len(s)) ]
  return s_list, lnP
   




def bnSample(model, order):
  """Draw a sample from a Bayes net model with given topo order """
  # TODO: fix
  x = [-1 for Xi in order]   # assumes 0..N ids
  # TODO: isBN, this need topo ordering of factors
  for i in order:
    # which factor is the conditional?  should be one with only earlier vars
    parents = factors[i].vars - [X[i]]
    #print str(i)+" has parents "+str(parents)+", values "+str([x[p] for p in parents])
    Pi = factors[i].condition2( parents, [x[p] for p in parents] )
    # if scalar (already sampled), save value (fwd sampling?)
    x[i], = Pi.sample()    # returns length-1 tuple; save value in config x
  #print "Sample from p(x): " + str( x )
  return x

