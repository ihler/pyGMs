"""
Weighted ordered graphical model:
A graphical model (GraphModel) augmented with an elimination order and weights (inverse powers) 
for that elimination, corresponding to an inference problem.
Always maintains factors is log form


Constructor:  model or factors, order, temperature / weights.

Order construction:  take existing order & temperature, convert to equivalence classes, then
  re-order among equivalence classes to reduce induced width?

Set order & temperature together?






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
from .factor import *
from .graphmodel import *

import time
from builtins import range

try:
  from itertools import izip
except:
  izip = zip
reverse_enumerate = lambda l: izip(range(len(l)-1, -1, -1), reversed(l))



def __wt_elim(f,w,pri):
    """Helper function for performing weighted elimination of a factor

       Args: Factor f; weight array w (one entry per variable in f); numpy array priority
    """
    elim_ord = np.argsort( pri[f.vars.labels] )
    tmp = f
    for i in elim_ord: tmp = tmp.lsePower([f.v[i]],1.0/w[i]);
    return float(tmp)   # should be a float, or a Factor([],float)


def calc_bounds( thetas, weights, pri):
    """Compute the GDD / Holder bound on log-factors [thetas] with weights [weights]"""
    return [__wt_elim(th,wt,pri) for th,wt in zip(thetas,weights)]

def calc_deriv(th,w,pri,match,Xi=None):
    """Compute the GDD / Holder bound gradient, for optimization"""
    elim_ord = np.argsort( [pri[x] for x in th.vars] )
    lnZ0 = th.copy()
    lnmu = 0.0;
    for i in elim_ord:          # run over v[i],w[i] in the given elim order
        lnZ1  = lnZ0.lsePower([th.v[i]],1.0/w[i])
        lnZ0 -= lnZ1;             # update lnmu += (lnZ0 - lnZ1)*(1.0/w[i])
        lnZ0 *= (1.0/w[i]);
        lnmu += lnZ0;             # TODO: save a copy by assigning = lnZ0 on 1st loop?    
        lnZ0 = lnZ1;              # then move to the next conditional
    lnmu.expIP()
    Hxi = 0.0
    if Xi is not None:
        keep = [x for x in th.vars if pri[x]>=pri[Xi]]
        forH = lnmu.marginal(keep) if len(keep) < th.nvar else lnmu
        Hxi = forH.entropy() - forH.sum([Xi]).entropy() if forH.nvar > 1 else forH.entropy()
    return lnmu.marginal(match), Hxi

def calc_deriv2(th,w,pri,match,Xi=None):
    """Hacked version of calc_deriv() for max variables in mmap problems"""
    elim_ord = np.argsort( [pri[x] for x in th.vars] )
    lnZ0 = th.copy()
    lnmu = 0.0;
    for i in elim_ord:          # run over v[i],w[i] in the given elim order
        if th.v[i] in match: lnmu += lnZ0; break   # TODO: hacky!!!
        lnZ1  = lnZ0.lsePower([th.v[i]],1.0/w[i])
        lnZ0 -= lnZ1;             # update lnmu += (lnZ0 - lnZ1)*(1.0/w[i])
        lnZ0 *= (1.0/w[i]);
        lnmu += lnZ0;             # TODO: save a copy by assigning = lnZ0 on 1st loop?    
        lnZ0 = lnZ1;              # then move to the next conditional
    return lnmu.maxmarginal(match), 0.0




class WOGraphModel(GraphModel):
  """Class implementing a weighted & ordered graphical model"""
  temp = 1e-3  # temperature for weighted max-elim where required

  def __init__(self, factorList=None, elimOrder=[], weights=[], copy=True, isLog=False):
    """Create a weighted, ordered graphical model object.  Maintains local copies of the factors.

       elimOrder : the elimination order for the model (list of variables)
       weights   : the elimination weights for each variable (list of floats)
         Note: while these can be initalized to [], they must be set before the model can be reasoned over; 
               elimOrder must be set before weights.
       copy = [True] False  : by default, the model makes its own internal copy of all factors
       isLog = True [False] : if true, the factors are interpreted as log-probability factors
    """
    super(WOGraphModel, self).__init__(factorList, copy=copy, isLog=isLog)
    self.toLog()                                                              # Ensure log-factors
    if self.factors[0].nvar > 1: self.addFactors([Factor([],0.0)],copy=False) # add constant factor if not present
    self.elimOrder = elimOrder
    self.weights = weights
    self.v_beliefs = [ Factor([v],1.0/v.states) for v in self.X ]


 
  @property
  def elimOrder(self):
    return self.__elimOrder
  @elimOrder.setter
  def elimOrder(self, elimOrder):
    self.__elimOrder = elimOrder
    self.priority = np.zeros((len(elimOrder),));
    self.priority[elimOrder] = np.arange(len(elimOrder),dtype=long);
    if len(elimOrder)==0: return   # blank order...
    # Build bucket lists: factors by earliest eliminated variable
    self.bucket = [ [] for v in self.X ]
    for f in self.factors:
      if f.nvar == 0: continue
      b = f.v[ np.argsort( self.priority[f.v.labels] )[0] ]   # 1st eliminated variable in f
      self.bucket[b].append(f)
      

    # Connect each factor to its parent:
    # TODO for f in self.factors: find & save factor containing all after v, or None

  @property
  def weights(self):
    return self.__weights
  @weights.setter
  def weights(self, weights):
    self.__weights = np.array(weights)
    self.fweights = { f: np.zeros((f.nvar,)) for f in self.factors }
    if len(weights)==0: return   # blank weights
    # TODO: FIX for zero weight, negative weight, etc.
    xweights = np.zeros((len(self.priority)))
    for f in self.factors:
      ord_f = np.argsort( self.priority[f.v.labels] )
      for x in ord_f[:-1]: self.fweights[f][x] = 1.0; xweights[f.v[x]] += 1.0
      for x in ord_f[-1:]: self.fweights[f][x] = self.temp; xweights[f.v[x]] += self.temp;
    # TODO: should "elect" a 1.0 weight?
    for f in self.factors:
      self.fweights[f] *= self.weights[f.v.labels].clip(min=self.temp) / xweights[ f.v.labels ]   # normalize the weights
 

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
          if self.isLog: constant += fc[0]            # if it's now a constant, just pull it out 
          else:          constant += np.log(fc[0])
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
      self.fweights[f] = np.array([self.temp])


  def costtogo(self, X ):
    ret = Factor([self.X[X]], 0.0)
    for f in self.factorsWith(X):
      w = self.fweights[f]
      elim_ord = np.argsort( self.priority[f.vars.labels] )
      tmp = f
      for i in elim_ord: tmp = tmp if f.v[i]==X else tmp.lsePower([f.v[i]],1.0/w[i])
      ret += tmp
    return ret

  # TODO: def mergeFactors( fa, fb ):

  # TODO: memory bound MBE-ization

  # TODO: change in memory of MBE-zation ???
  
  # TODO: MBE-ize the model: ensure all factors have parents or are singletons; shift costs.  



  @staticmethod
  def update_weights(weights,idx,dW,stepW):   # TODO only works for positive weights
      wtot = 0.0
      for j,wt,dw in zip(idx,weights,dW): wt[j] *= np.exp( - stepW * wt[j] * dw ); wtot += wt[j];
      for j,wt,dw in zip(idx,weights,dW): wt[j] /= wtot;

  def maxsumdiff(self,thetas,weights,match,Xi):
      dT = list( calc_deriv2(th,wt,self.priority,match,None)[0] for th,wt in zip(thetas,weights) )
      avg = 0.0
      for dt in dT: avg += dt 
      self.v_beliefs[Xi] = avg.exp()
      avg /= len(dT)
      for th,dt in zip(thetas,dT): th -= dt; th += avg;
      maxval = avg.max()
      bounds = [maxval for th in thetas]
      return bounds

  def armijo(self, thetas,weights,Xi,steps,threshold=1e-4,direction=+1, optTol=1e-8,progTol=1e-8):
      import copy
      bounds = [0.0 for f in thetas]  # NOTE: assumes bounds already pulled out!
      f0,f1 = None, sum(bounds) #calc_bounds(thetas,weights,pri))    # init prev, current objective values
      match = reduce(lambda a,b: a&b, [th.vars for th in thetas], thetas[0].vars) 
      if np.all(self.weights[match.labels]==0.0): return self.maxsumdiff(thetas,weights,match,Xi);
      # TODO: save magnitude of gradient, for use in scheduling?
      idx = [th.v.index(Xi) for th in thetas] if Xi is not None else []  # find location of Xi in var/weight vectors
      newweights = copy.deepcopy(weights) if Xi is not None else weights # copy weights if updated TODO SIMPLIFY
      for s in range(steps):
        # compute gradients dPhi/dTheta, dPhi/dW  (wrt parameters, weights):
        dT,dW = (list(l) for l in zip(*[calc_deriv(th,wt,self.priority,match,Xi) for th,wt in zip(thetas,weights)]))
        if Xi is not None: self.v_beliefs[Xi] = dT[0].marginal([Xi])
        for dt in dT[1:]: dt -= dT[0]; dt *= -1;
        dT[0] = -sum(dT[1:])
        if Xi is not None:
          Hbar = sum([wt[j]*dw for j,dw,wt in zip(idx,dW,weights)])
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
        for j in range(10):
          newthetas = [th+dt for th,dt in zip(thetas,dT)]   # step already pre-multiplied
          if Xi is not None: WOGraphModel.update_weights( newweights, idx, dW, step );
          bounds = calc_bounds(newthetas,newweights,self.priority)
          f1 = sum(bounds)
          #print "  ",f0," => ",f1, "  (",f0-f1,' ~ ',stepsize*threshold*gradnorm,")"
          if (f0 - f1)*direction > step*threshold*L2:       # if armijo "enough improvement" satisfied
            for th,nth in zip(thetas,newthetas): np.copyto(th.t,nth.t)  # rewrite tables
            for j,wt,w2 in zip(idx,weights,newweights): wt[j] = w2[j];
            break;
          else:                                             # ow, back off 
            step *= 0.5;
            if step*L0 < progTol: return bounds             # if < progTol => no progress possible
            for dt in dT: dt *= 0.5
      return bounds


  def init(self):
    # check if uninitialized and flag as initialized?
    bounds = calc_bounds(self.factors, [self.fweights[f] for f in self.factors], self.priority)
    for j,f in enumerate(self.factors): f -= bounds[j]   # remove scalar bounding constant
    phi_w = sum(bounds)
    self.factors[0][0] += phi_w                             # and put it in the scalar factor


  def update(self, stopIter=100, stopTime=inf, stopTol=0.0, stepInner=5, verbose=False):
    # check if uninitialized and initialize if not?
    self.init()
    phi_w = self.factors[0][0]
    start_time = time.time()
    if verbose: print "Iter 0: ", phi_w
    for t in xrange(1,stopIter+1):               # for each iteration:
        # Update each variable in turn:
        for Xi in self.elimOrder:                # for each variable,  TODO: use elim order
            with_i = self.factorsWith(Xi)
            if len(with_i) <= 1: continue;
            weight_i = [self.fweights[th] for th in with_i]
            bounds = self.armijo(with_i,weight_i,Xi, 5, 0.01, float(self.weights[Xi] >= 0.0) )
            for f,bd in zip(with_i,bounds): f-=bd  # modify the factors to pull out bound
            self.factors[0][0] += sum(bounds)      # and store in constant term
        #
        # Update the upper bound, print, and check for convergence
        prev,phi_w = phi_w,self.factors[0][0]
        if verbose: print "[{}] Iter {} : {}".format(time.time()-start_time,t,phi_w);
        if abs(prev - phi_w) < stopTol: break
    return phi_w





