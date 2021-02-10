"""
Weighted graphical model:
A graphical model (GraphModel) augmented with an elimination order and weights (temperatures) for each Xi 
Corresponds to an inference problem, Phi_w = \log [ ... \wsum_{x_i} ... \exp[ \sum_r \theta_r(X_r) ] ... ]
Always maintains factors in log form (theta_r)

Constructor takes factors, order (default 1..N), weights (default 1.0)
Can set order, weights manually
Can "reorder" -- find new elimination order that preserves problem definition (e.g., only switch
the order of variables with equal weights).

Supports incremental construction
Supports mini-bucket construction, heuristics, and importance sampling

class WOGraphModel:
  # attributes:
  #   elimOrder[i]
  #   priority[Xj] = i if elimOrder[i]=Xj
  #   bucket[i] = [nodei1 nodei2 ... ]
  #   matchlist[i] = [matchi1 matchi2 ...]  
  #
  #   class Node:
  #     clique = VarSet
  #     theta = factor (or list of factors?)
  #     weight = float
  #     parent = ref, children = [refs...]  or index?
  #     msgFwd, msgBwd = factor

"""

"""
Notes: expected usage / functions
  constructor: build from factors or log factors
  find elimination order based on "priority" (0/1 for mmap)
  set weights for each variable (0/1 for mmap); may be before elim order!
  evaluate current bound (does work if first time?)
  [what happens if we update a factor?]
  tighten bound with message passing [how scheduling?  details?  epsilon bound?]
  merge two cliques if desired, updating their weights & bound info
    [preserve "mini-bucket-ness" if desired?  do/don't build functions as we go?]
  find a "good" variable [access beliefs or similar info?]
  bound conditioned subproblems for a variable
  condition on a variable's value
"""

from .factor import *
from .graphmodel import *

import time
from builtins import range

try:
  from itertools import izip
except:
  izip = zip
reverse_enumerate = lambda l: izip(range(len(l)-1, -1, -1), reversed(l))


one = np.float64(1.0)   # useful for 1/w = inf if w=0



def PhiW(f,w,pri,exclude=[],temp=0.0):
    """Helper function for performing weighted elimination of a factor

       Args: Factor f; weight array w (one entry per variable in f); numpy array priority
    """
    if f.nvar == 0: return float(f)
    elim_ord = np.argsort( pri[f.vars.labels] )    # TODO: sort each time, or store somewhere?
    tmp = f
    for i in elim_ord: 
      if f.v[i] not in exclude: tmp = tmp.lsePower([f.v[i]],one/(temp if w[i]==0 else w[i]));   # TODO: runs of vars with equal temp?
    return tmp   

def d_PhiW(th,w,pri,match,Xi=None,temp=0.0):
    """Compute the GDD / Holder bound gradient, for optimization"""
    elim_ord = np.argsort( [pri[x] for x in th.vars] )
    lnZ0 = th.copy()
    lnmu = 0.0;
    for i in elim_ord:          # run over v[i],w[i] in the given elim order
        lnZ1  = lnZ0.lsePower([th.v[i]],one/(temp if w[i]==0 else w[i]))
        lnZ0 -= lnZ1;             # update lnmu += (lnZ0 - lnZ1)*(1.0/w[i])
        lnZ0 *= (one/(temp if w[i]==0 else w[i]));
        lnmu += lnZ0;             # TODO: save a copy by assigning = lnZ0 on 1st loop?    
        lnZ0 = lnZ1;              # then move to the next conditional
    lnmu.expIP()
    Hxi = 0.0
    if Xi is not None:
        keep = [x for x in th.vars if pri[x]>=pri[Xi]]
        forH = lnmu.marginal(keep) if len(keep) < th.nvar else lnmu
        Hxi = forH.entropy() - forH.sum([Xi]).entropy() if forH.nvar > 1 else forH.entropy()
    return lnmu.marginal(match), Hxi




class WeightedModel(GraphModel):
  """Class implementing a weighted & ordered graphical model"""
  temp = 1e-3  # temperature for weighted max-elim where required

  def __init__(self, factorList=None, elim_order=[], weights=1.0, copy=True, isLog=False):
    """Create a weighted, ordered graphical model object.  Maintains local copies of the factors.

       elimOrder : the elimination order for the model (list of variables)
       weights   : the elimination weights for each variable (list of floats)
         Note: while these can be initalized to [], they must be set before the model can be reasoned over; 
               elimOrder must be set before weights.
       copy = [True] False  : by default, the model makes its own internal copy of all factors
       isLog = True [False] : if true, the factors are interpreted as log-probability factors
    """
    super(WeightedModel, self).__init__(factorList, copy=copy, isLog=isLog)
    self.toLog()                                                              # Ensure log-factors
    if self.factors[0].nvar >= 1: self.addFactors([Factor([],0.0)],copy=False) # add constant factor if not present
    self.elim_order = elim_order
    self.weights = weights
    self.v_beliefs = [ Factor([v],1.0/v.states) for v in self.X ]


  def _bucket_of(self,f):
    """Find bucket of factor f = 1st eliminated variable in f's scope"""
    return self._elim_order[min( self.priority[f.v.labels] )]   # TODO: test / check correctness
    #return f.v[ np.argsort( self.priority[f.v.labels] )[0] ]

  def _merge(self,f1,f2):
    """Merge two factors in the model, combining their weights"""
    w1,w2 = self.fweights[f1], self.fweights[f2]
    f = f1+f2
    w = np.zeros((f.nvar,))
    for i,X in enumerate(f.vars):
      if X in f1.vars: w[i] += w1[f.index(X)]
      if X in f2.vars: w[i] += w2[f.index(X)]
    self.removeFactors([f1,f2])
    self.fweights.pop(f1)
    self.fweights.pop(f2)
    self.addFactors([f])
    self.fweights[f] = w


  @property
  def elim_order(self):
    return self._elim_order
  @elim_order.setter
  def elim_order(self, elim_order):
    self._elim_order = elim_order
    self.priority = np.zeros((len(self.X),),dtype=long);
    self.priority[elim_order] = np.arange(len(elim_order),dtype=long);
    if len(elim_order)==0: return   # blank order...
    # Build bucket lists: factors by earliest eliminated variable
    self.bucket = { X:[] for X in self.X }
    for f in self.factors:
      if f.nvar > 0: self.bucket[ self._bucket_of(f) ].append(f)
      
    # Connect each factor to its parent:
    #self.parent = { f:None for f in self.factors }
    # TODO for f in self.factors: find & save factor containing all after v, or None

  @property
  def weights(self):
    return self._weights
  # TODO: doesn't work (too easy to change individual entries); use set function or something to expose the issue
  @weights.setter
  def weights(self, weights):
    if type(weights) is float: weights = np.zeros((self.nvar,))+weights;
    self._weights = np.array(weights)
    self.fweights = { f: np.zeros((f.nvar,)) for f in self.factors }
    if len(weights)==0: return   # blank weights
    for X in self.X: 
      withX = self.factorsWith(X, copy=False)
      if not len(withX): continue;
      if (self._weights[X] == -1): 
        for f in withX[-1:]: self.fweights[f][f.v.index(X)] = 2.0 if len(withX)>1 else 1.0
        for f in withX[:-1]: self.fweights[f][f.v.index(X)] = -1.0 / (len(withX)-1)
      elif (self._weights[X] == 0.0):
        for f in withX: self.fweights[f][f.v.index(X)] = 0.0
      elif (self._weights[X] == 1.0):
        fUse = withX
        #fUse = [f for j,f in enumerate(withX) if max(self.priority[f.v.labels])!=self.priority[X]]
        #if len(fUse)==0: fUse.append(withX[0])
        for f in fUse: self.fweights[f][f.v.index(X)] = 1.0/len(fUse)
      else: raise ValueError('Unknown weight {} for {}.format(self._weights[X],X)')
      
    ## TODO: FIX for zero weight, negative weight, etc.
    #xweights = np.zeros((len(self.priority),))
    #for f in self.factors:
    #  ord_f = np.argsort( self.priority[f.v.labels] )
    #  for x in ord_f[:-1]: self.fweights[f][x] = 1.0; xweights[f.v[x]] += 1.0
    #  for x in ord_f[-1:]: self.fweights[f][x] = self.temp; xweights[f.v[x]] += self.temp;
    ## TODO: should "elect" a 1.0 weight?
    #for f in self.factors:
    #  self.fweights[f] *= self.weights[f.v.labels].clip(min=self.temp) / xweights[ f.v.labels ]   # normalize the weights

  # TODO: override addFactors, etc? 
 
  
  def condition(self, evidence):
    """Condition / clamp the graphical model on a partial configuration (dict) {Xi:xi,Xj:xj...}"""
    # TODO: optionally, ensure re-added factor is maximal, or modify an existing one (fWithAll(vs)[-1]?)
    if len(evidence)==0: return
    for v,x in evidence.items():
      constant = 0.0
      for f in self.factorsWith(v):
        self.removeFactors([f])
        fc = f.condition({v:x})
        b = f.v[ np.argsort( self.priority[f.v.labels] )[0] ]  # remove the factor from its bucket
        self.bucket[b].pop( self.bucket[b].index(f) )
        # TODO: fix up parent structure ...
        if (fc.nvar == 0):
          constant += fc[0] if self.isLog else np.log(fc[0])  # if it's now a constant, just pull it out 
        else:
          self.addFactors([fc],copy=False)                       # otherwise add the factor back to the model
          bc= fc.v[ np.argsort( self.priority[fc.v.labels] )[0] ]  # add new factor to its bucket
          self.bucket[bc].append(fc)
          i = f.v.index(v)                            # copy over weights for new factor
          self.fweights[fc] = self.fweights[f][np.arange(f.nvar)!=i]
        self.fweights.pop(f)                          # finally, remove old factor weights 
      # add delta f'n factor for each variable, and distribute any constant value into this factor
      if self.isLog: f = Factor([self.X[v]],-inf); f[x] = constant
      else:          f = Factor([self.X[v]],0.0); f[x] = np.exp(constant)
      self.addFactors([f],copy=False)
      self.fweights[f] = np.array([np.abs(self._weights[v])])   ## TODO: abs? 1/0/-1?


  def costtogo(self, X ):
    bounds = [PhiW(f,self.fweights[f],self.priority,exclude=[X]) for f in self.factorsWith(X,copy=False)]
    return sum(bounds)

  # TODO: def mergeFactors( fa, fb ):

  # TODO: memory bound MBE-ization

  # TODO: change in memory of MBE-zation ???
  
  # TODO: MBE-ize the model: ensure all factors have parents or are singletons; shift costs.  


  #def update_weights(thetas,weights,match,Xi,dW,stepW):   # TODO only works for positive weights
  #    if self._weights[Xi] == 0: return
  #    if self._weights[Xi] > 0: 
  #      wtot = 0.0
  #      for j,wt,dw in zip(idx,weights,dW): wt[j] *= np.exp( - stepW * wt[j] * dw ); wtot += wt[j];
  #      for j,wt,dw in zip(idx,weights,dW): wt[j] /= wtot;

  @staticmethod
  def update_weights(weights,idx,dW,stepW,direction):   # TODO only works for positive weights
      wtot = 0.0
      Hbar = 0.0
      #wtot_initial = sum([wt[j] for j,wt in zip(idx,weights)])
      if (direction > 0):
        #Hbar = sum([wt[j]*dw for j,dw,wt in zip(idx,dW,weights)]) / sum([wt[j] for j,wt in zip(idx,weights)])
        for j,wt,dw in zip(idx,weights,dW): wt[j] *= np.exp( - stepW * wt[j] * (dw-Hbar) ); wtot += wt[j];
        for j,wt,dw in zip(idx,weights,dW): wt[j] /= wtot;
      else:
        #print([wt[j] for j,wt,dw in zip(idx,weights,dW)])
        ipos = [i for i,j,wt in zip(range(len(idx)),idx,weights) if wt[j]>0] 
        if len(ipos)!=1: raise ValueError('{} positive weights for lower bound?'.format(len(ipos)))
        ipos = ipos[0]
        #Hbar = dW[ipos]
        for i,j,wt,dw in zip(range(len(idx)),idx,weights,dW): 
          if i!=ipos: wt[j] *= np.exp( - direction * stepW * wt[j] * (dw-Hbar) ); wtot += wt[j];
        weights[ipos][idx[ipos]] = 1 - wtot
        #print([wt[j] for j,wt,dw in zip(idx,weights,dW)])
        #print('===')


  def maxsumdiff(self,thetas,weights,match):
      bounds = [PhiW(th,wt,self.priority,exclude=match) for th,wt in zip(thetas,weights)]
      avg = sum(bounds)/len(bounds)
      for th,bd in zip(thetas,bounds): th -= bd; th += avg;
      maxval = avg.max()
      bounds = [maxval for th in thetas]
      for X in match: 
        self.v_beliefs[X] = (1000*(avg.maxmarginal([X])-maxval)).expIP();  
        self.v_beliefs[X] /= self.v_beliefs[X].sum()
        # TODO: do we want this? or deltas? or nothing?
      return bounds

  def closed_form_ok(self,thetas,weights,match):
      if any(self._weights[match.labels]!=0): return False
      earliest = min(self.priority[match.labels])
      for th,wt in zip(thetas,weights):
        if any( (self.weights[th.v.labels]!=0) & (self.priority[th.v.labels]>earliest) ): return False
      return True

  def armijo(self, thetas,weights,Xi,steps,threshold=1e-4,direction=+1, optTol=1e-8,progTol=1e-8):
      import copy
      bounds = [0.0 for f in thetas]  # NOTE: assumes bounds already pulled out!
      #print("Update pass for ",Xi)
      # TODO: return something if thetas is length 1?
      f0,f1 = None, sum(bounds)                                          # init prev, current objective values
      match = reduce(lambda a,b: a&b, [th.vars for th in thetas], thetas[0].vars) 
      if self.closed_form_ok(thetas,weights,match): return self.maxsumdiff(thetas,weights,match)
      idx = [th.v.index(Xi) for th in thetas] if Xi is not None else []  # find location of Xi in var/weight vectors
      newweights = copy.deepcopy(weights) if Xi is not None else weights # copy weights if updated TODO SIMPLIFY
      for s in range(steps):
        # compute gradients dPhi/dTheta, dPhi/dW  (wrt parameters, weights):
        dT,dW = (list(l) for l in zip(*[d_PhiW(th,wt,self.priority,match,Xi,self.temp) for th,wt in zip(thetas,weights)]))
        if Xi is not None: a=np.argmax([wt[j] for j,wt in zip(idx,weights)]); self.v_beliefs[Xi] = dT[a].marginal([Xi]);
        for dt in dT[1:]: dt -= dT[0]; dt *= -1;
        dT[0] = -sum(dT[1:])
        if Xi is not None and self._weights[Xi]!=0.0:  #TODO needed?
          ipos = [i for i,j,wt in zip(range(len(idx)),idx,weights) if wt[j]>0]
          if len(ipos)!=1: # TODO: if all same sign, do this
            Hbar = sum([wt[j]*dw for j,dw,wt in zip(idx,dW,weights)]) / sum([wt[j] for j,wt in zip(idx,weights)])
          else:            # if exactly one positive, use this form
            Hbar = dW[ipos[0]]     
          for j in range(len(dW)): dW[j] -= Hbar
        # Compute gradient norms:
        L0,L1,L2 = zip(*[ (d.max(),d.sum(),(d*d).sum()) for dt in dT for d in [dt.abs()]])
        L0,L1,L2 = max(L0),sum(L1)+1e-300,sum(L2)
        L0,L1,L2 = max(L0,max(abs(dw) for dw in dW)), L1+sum(abs(dw) for dw in dW), L2+sum(dw*dw for dw in dW)
        if L0 < optTol: return bounds                       # if < optTol => local optimum  

        step = min(1.0, 1.0/L1) if f0 is None else min(1.0, direction*(f0-f1)/L1)
        step = step if step > 0 else 1.0
        f0 = f1;                                            # update "old" objective value
        for dt in dT: dt *= direction*step;                 # premultiply step size into dT
        for j in range(20):
          newthetas = [th+dt for th,dt in zip(thetas,dT)]   # step already pre-multiplied
          if Xi is not None and self._weights[Xi]!=0: WeightedModel.update_weights( newweights, idx, dW, step, direction );
          bounds2 = [PhiW(th,wt,self.priority) for th,wt in zip(newthetas,newweights)]
          f1 = sum(bounds2)
          #print("  ",f0," => ",f1, "  (",f0-f1,' ~ ',step*threshold*L2,") ",(f0 - f1)*direction > step*threshold*L2)
          if (f0 - f1)*direction > step*threshold*L2:       # if armijo "enough improvement" satisfied
            for th,nth in zip(thetas,newthetas): np.copyto(th.t,nth.t)  # rewrite tables
            for j,wt,w2 in zip(idx,weights,newweights): wt[j] = w2[j];
            bounds = bounds2
            break;
          else:                                             # ow, back off 
            step *= 0.5;
            if step*L0 < progTol: return bounds             # if < progTol => no progress possible
            for dt in dT: dt *= 0.5
            f1 = f0
      return bounds


  def init(self):
    # TODO check if uninitialized and flag as initialized?
    bounds = [PhiW(f,self.fweights[f],self.priority) for f in self.factors]
    #print(bounds)
    for j,f in enumerate(self.factors): f -= bounds[j]   # remove scalar bounding constant
    phi_w = sum(bounds)
    self.factors[0][0] += phi_w                             # and put it in the scalar factor


  def update(self, stopIter=100, stopTime=inf, stopTol=0.0, stepInner=5, verbose=False):
    # check if uninitialized and initialize if not?
    self.init()
    phi_w = self.factors[0][0]
    start_time = time.time()
    if verbose: print("Iter 0: ", phi_w)
    for t in range(1,stopIter+1):                 # for each iteration:
        # Update each variable in turn:
        for Xi in self.elim_order:                # for each variable,  TODO: use elim order
            with_i = self.factorsWith(Xi)
            if len(with_i) <= 1: continue;
            weight_i = [self.fweights[th] for th in with_i]
            #print(weight_i)
            bounds = self.armijo(with_i,weight_i,Xi, 5, 0.01, 2*float(self.weights[Xi] >= 0.0)-1 )
            for f,bd in zip(with_i,bounds): f-=bd  # modify the factors to pull out bound
            self.factors[0][0] += sum(bounds)      # and store in constant term
        #
        # Update the upper bound, print, and check for convergence
        prev,phi_w = phi_w,self.factors[0][0]
        if verbose: print("[{}] Iter {} : {}".format(time.time()-start_time,t,phi_w))
        if abs(prev - phi_w) < stopTol: break
    return phi_w


  def reorder(self, *args,**kwargs):
    """Find a new elimination order that respects the current weighted elimination task"""
    priority = [0 for X in self.X]
    wt = float('nan')
    pri= 0
    for X in self.elim_order: 
      if self._weights[X]==wt: priority[X]=pri
      else: pri = pri+1; wt = self._weights[X]; priority[X]=pri;
    elim,score = eliminationOrder(self,*args,priority=priority,**kwargs)
    self.elim_order = elim  # TODO: doesn't work well with cutoff operation!


