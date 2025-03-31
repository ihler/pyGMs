"""
montecarlo.py

Defines several Monte Carlo and MCMC routines for approximate inference in graphical models

Version 0.3.0 (2025-03-31)
(c) 2015 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import numpy as np
#import time as time
from time import time
from sortedcontainers import SortedSet;
from builtins import range
from functools import reduce

from pyGMs.factor import *
from pyGMs.graphmodel import *


####### Basic sampling #########################
#
# sample( factors ) -- in sequence, draw all vars not yet sampled, conditioned on sampled values; also return q(x)
#
#   sample => bnSample for bn with no evidence?
#   conversion of mrf to bn for sampling? computation of R for rejection sampling?
#   
# 
# importance sample:  sample(fs) => x,q ; value(x) => p  => x, w=p/q
#
# rejection sample: sample(fx) => x,q ; reject if r > p/q*R  where q*R > p for all x   (shortcut?)
#   bn version? reject if Xe != evid?  or just use generic version w/ q=bn, p=bn.condition(evid), R=1?
# 
# likelihood weighting => importance sample?  likelihood reweighting?
#   draw from bn.condition(evid)? how to 
#
# gibbs sample: 
#
#
# fwdSample (?) - in a BN, sample each factor in sequence; weight by evidence  => = IS w/ conditionals
#

#
# Basic sampling function interface:
#   x,logp = sample();    # return configuration "x" and log p(x), the log-probability of drawing that sample
# For Markov chain Monte Carlo:
#   xb,logp = sample(xa); # return sampled transition xa -> xb and log p(xa->xb), the probability of that transition 
#

# Sampling with a time limit: [model.sample() for i in gm.timelimit(5.0)]
# 
# Some sampling methods do not produce a (log-) weight, e.g. GibbsSampler; then:
#   gibbs_single = lambda: (gibbs.sample(),0.0);  gibbs_single() # produces single sample with log-weight 0.
# (maybe TODO: implement gibbs sampler as an iterator?)

# Basic query interface:
#   Q = Query( f )                   # to compute expectation of f(x): E_p[f] 
#   Q = Query( [f1, f2, ...] )       # to compute E_p[fi] for each fi in the list
#   Q = QueryMarginals( factorlist ) # to compute p(x_a) for each factor f_a(x_a) in the list
#
# A query is an object Q=Query(..) with:
#   Q() or Q[i]   : return the current estimate, or ith estimate from a list
#   Q.update(x,w) : update the expectations' estimates by observing state x with weight w
#   Q.wvar()      : return the empirical variance of the weights encountered (optional)


def neff(weights, isLog=False):
    """Compute the number of effective samples from the weights of a data sample.
      weights (list[float]) : list of weights, one per data point
      isLog (bool, default False) : if True, normalized exp(weights) is used instead 
            (useful for sampling functions that return log-probabilities or log-weights)
    """
    if isLog:
        logw = np.asarray(weights)
        mx = logw.max(); 
        w = np.exp(logw-mx);
    else: 
        w = np.asarray(weights)
    return (np.sum(w)**2)/np.sum(w**2)



class EmpiricalStatistics(object):
    """Keep track of the empirical expectations of a function / set of functions, given a stream of (weighted) data.
    Ex:
      stats = EmpiricalStatistics( lambda x: x[1]/(x[0]+1) )  
        # estimates a scalar expectation, E[ x1/x0 ], from vector data [(x0,x1,x2,...) ...
      stats = EmpiricalStatistics( {i:partial(lambda j,x: x[j], i) for i in range(n)} )
        # estimates a collection of expectations, E[xi] for each i=0..n-1

    Interface functions include:
      stats()  : return the current estimate(s)
      stats[i] : return the ith estimate of a list of functions, or function with key i for a dict of functions
      stats.update(x,w) : update the estimates after observing state "x" with weight "w"
      stats.reset()     : reset / re-initialize estimates
    It may also have
      stats.nsamples  : total number of samples (calls to update)
      stats.wtot      : total of weights seen during calls to update
      stats.neff      : number of "effective" samples
      stats.wvar      : variance of weight values
    """
    def __init__(self, functions):
        self.isList = True
        if (not hasattr(functions,"__getitem__")):
            functions = [functions]
            self.isList = False
        self.functions = functions
        self.reset()
        # TODO: should probably be in log-weight domain

    def reset(self):
        self.sums  = [0.0] * len(self.functions)
        self.nsamples = 0.0
        self.wtot  = 0.0  # save weight total
        self.w2tot = 0.0  # save weight^2 total

    def update(self,x,w):
        for i,f in enumerate(self.functions):
            self.sums[i] += f(x)
        self.nsamples += 1
        self.wtot += w
        self.w2tot+= w**2

    def __getitem__(self,i):              # TODO: list vs dict of functions?
        return self.sums[i]/self.wtot

    def __call__(self):
        to_return = [ s / self.wtot for s in self.sums ]
        if not self.isList: to_return = to_return[0]
        return to_return

    @property
    def wvar(self):
        return (self.w2tot - self.wtot**2)/self.nsamples

    @property
    def neff(self):
        return self.wtot**2 / self.w2tot



class EmpiricalMarginals(EmpiricalStatistics):
    """Specialized empirical statistics object for estimating the marginal probabilities of variable sets
    Ex: 
      stats = EmpiricalMarginals( [gm.VarSet([Xi]) for Xi in model.X] ) # estimate marginal p(x_i) for each xi 
      stats = EmpiricalMarginals( [f.vars for f in model.factors] ) # estimate marginal p(x_a) for each factor fa(xa)
    """
    def __init__(self, factorlist):
        self.sums = [ Factor(f.vars,0.0) for f in factorlist ]
        self.nsamples = 0.0
        self.wtot = 0.0
        self.w2tot= 0.0

    def update(self,x,w):
        for mu in self.sums:
            mu[ tuple(x[v] for v in mu.vars) ] += w
        self.nsamples += 1
        self.wtot += w
        self.w2tot += w**2

    #def __getitem__(self,i):
    #    return self.marginals[i]/self.wtot
    #
    #def __call__(self):
    #    return [ mu / self.wtot for mu in self.marginals ]
    #
    # TODO: should probably just inherit from Query
    #@property
    #def wvar(self):
    #    return (self.w2tot - self.wtot**2)/self.nsamples
    #
    #@property
    #def neff(self):
    #    return self.wtot**2 / self.w2tot







"""


class Query(object):
    ""Defines a Monte Carlo "query" object, for estimation of various quantities from sequences of (weighted) states.
       Q = Query( f )   # function f(x), estimate expectation E_p[ f(x) ]
       Q = Query( [f1,f2,...] )   # functions fi(x), estimate each expectation

       An object with a query interface should include at least:
           Q()  : return the current estimate(s)
           Q[i] : return the ith estimate (if a list of estimates)
           Q.update(x,w) : update the estimates after observing state "x" with weight "w"
           Q.reset()     : reset / re-initialize estimates
       It may also have
           Q.nsamples  : total number of samples (calls to update)
           Q.wtot      : total of weights seen during calls to update
           Q.neff      : number of "effective" samples
           Q.wvar      : variance of weight values
    ""
    def __init__(self, functions):
        self.isList = True
        if (not hasattr(functions,"__getitem")): 
            functions = [functions]
            self.islist = False
        self.functions = functions
        self.sums  = [0.0] * len(functions)
        self.nsamples = 0.0
        self.wtot  = 0.0  # save weight total
        self.w2tot = 0.0  # save weight^2 total
        # TODO: should probably be in log-weight domain

    def update(self,x,w):
        for i,f in enumerate(self.functions):
            self.sums[i] += f(x)
        self.nsamples += 1
        self.wtot += w
        self.w2tot+= w**2

    def __getitem__(self,i):
        return self.sums[i]/self.wtot

    def __call__(self):
        to_return = [ s / self.wtot for s in self.sums ]
        if not self.isList: to_return = to_return[0]
        return to_return

    @property
    def wvar(self):
        return (self.w2tot - self.wtot**2)/self.nsamples

    @property
    def neff(self):
        return self.wtot**2 / self.w2tot
"""

###################################################### 
"""
class QueryMarginals(Query):
    ""Specialized Monte Carlo "query" object for marginal probabilities of factors
       Q = QueryMarginals( factorlist ) # estimate marginal p(x_a) for each factor f_a(x_a) in list
    ""
    def __init__(self, factorlist):
        self.sums = [ Factor(f.vars,0.0) for f in factorlist ]
        self.nsamples = 0.0
        self.wtot = 0.0
        self.w2tot= 0.0

    def update(self,x,w):
        for mu in self.sums:
            mu[ tuple(x[v] for v in mu.vars) ] += w
        self.nsamples += 1
        self.wtot += w
        self.w2tot += w**2

    #def __getitem__(self,i):
    #    return self.marginals[i]/self.wtot
    #
    #def __call__(self):
    #    return [ mu / self.wtot for mu in self.marginals ]
    #
    # TODO: should probably just inherit from Query
    #@property
    #def wvar(self):
    #    return (self.w2tot - self.wtot**2)/self.nsamples
    #
    #@property
    #def neff(self):
    #    return self.wtot**2 / self.w2tot

"""

"""
class QuerySamples(Query):
    ""Save a circular buffer of "keep" samples, inserting every "stride" MCMC steps.
    ""
    def __init__(self, keep=100, stride=10):
        self.samples = [ None for i in range(keep) ]
        self.i = 0
        self.n = 0
        self.stride = stride
    def __getitem__(self,i):
        return self.samples[(self.i+i)%len(self.samples)]
    def __call__(self): 
        return [self.samples[(i+self.i)%len(self.samples)] for i in range(len(self.samples)) ]
    def update(self,x,w):
        if (self.n % self.stride == 0):
            self.samples[self.i] = x
            self.i += 1; self.i = self.i % len(self.samples)
        self.n += 1
"""


def GibbsSampling( model, query, state=None, stopSamples=1, stopTime = inf ):
    """Gibbs sampling procedure for discrete graphical model "model" with query object "query"
    """
    state = state if state is not None else [np.random.randint(Xi.states) for Xi in model.X]
    # TODO: timing check
    for j in range(stopSamples):
        # TODO if out of time, break
        for Xi in model.X:
            p = Factor([],1.0)
            for f in model.factorsWith(Xi,False):
                cvar = f.vars - [Xi]
                p *= f.condition2( cvar, [state[v] for v in cvar] )
            p /= p.sum()
            state[Xi] = p.sample()[0]
        query.update(state, 1.0)
    return query


def GibbsSampling2( model, query, state=None, stopSamples=1, stopTime = inf ):
    """Gibbs sampling procedure for discrete graphical model "model" with query object "query"
    """
    if state is None: state = [ np.random.randint(v.states) for v in model.X]
    stateMap = { v:state[v] for v in model.X }
    j = 0; stopTime += time();
    while (j < stopSamples) and (time() < stopTime):
        for Xi in model.X:
            stateMap.pop(Xi)
            if model.isLog:
              p = reduce(lambda f,g:f+g, (f.condition(stateMap) for f in model.factorsWith(Xi,False)) )
              p -= p.lse();
              p.expIP()
            else: 
              p = reduce(lambda f,g:f*g, (f.condition(stateMap) for f in model.factorsWith(Xi,False)) )
              p /= p.sum()
            state[Xi] = p.sample()[0]
            stateMap[Xi] = state[Xi]
        j += 1 
        query.update(state, 1.0)
    return query


def GibbsSamplingBlock( model, query, state=None, blocks=None, stopSamples=1, stopTime = inf ):
    """Gibbs sampling procedure for discrete graphical model "model" with query object "query"
    """
    if state is None: state = [ np.random.randint(v.states) for v in model.X]
    if isinstance(state,tuple): state = list(state)
    if blocks is None: blocks = [ [x] for x in model.X ]   # basic single-var Gibbs
    j = 0; stopTime += time()
    while (j < stopSamples) and (time() < stopTime):
        for b in blocks:
            Pb = np.prod( [f.condition({v:state[v] for v in f.vars-b}) for f in model.factorsWithAny(b)] )
            xb = Pb.sample()
            for i,xi in zip(b,xb): state[i]=xi
        j += 1
        query.update(state, 1.0)
    return query




#
# TODO: change all to "object based" (e.g. GibbsSampler); change all to use list[dict] types for states (?)
#  don't forget to copy init_states (or provide flag for in-place updates)
#  

class GibbsSampler:
    """Implementation of Gibbs sampling for discrete graphical models """
    def __init__(self, model, chains=1, steps=1, init_states=None, blocks=None, in_place=False):
        """Gibbs sampling object constructor
        Args:
          chains (int) : number of (simultaneous) Markov chains to maintain
          steps (int) : default number of steps per call to sample()
          init_states (None,list[dict]) : initial states for the chain if specified; len(init_states)=chains
          blocks (None,list[varset]) : list of variables to sample simultaneously at each step (brute force)
            default: None => use model.X, i.e., sample each variable individually in sequence. 
            Note: union(blocks) should cover all variables to be updated
          in_place (bool) : update the object "init_states" in-place; default False (copies init_states)
        """
        ## TODO: if init_states is list of dicts, convert to list of tuples/lists/nparray (d2t(init_states)?)
        ## TODO: specialty behavior if chains is None? (single chain, not in list form?)
        ## TODO: no in-place (doesn't make sense? just copy & copy out too)
        self.model = model
        if init_states is None: init_states = [ list(np.random.randint(v.states) for v in self.model.X) for c in range(chains)]
        elif not in_place: init_states = [ list(x) for x in init_states ]    # copy state vectors
        #if init_states is None: init_states = [ {v:np.random.randint(v.states) for v in self.model.X} for c in range(chains)]
        #elif not in_place: init_states = [ {i:x[i] for i in x} for x in init_states ]    # copy dicts
        if blocks is None: blocks = [ [x] for x in self.model.X ]   # basic single-var Gibbs
        if len(init_states)!=chains: raise ValueError(f'Wrong number of initial states {len(init_states)} specified for {c} chains!')
        self.states = init_states 
        self.blocks = blocks
        self.steps = steps

    #@property
    #def states(self): return self.states  # TODO: copy to avoid accidental changes? Setter?
    # state if chains is None?

    ## TODO: __iter__ and __next__ to perform single-step sampling?

    def sample(self, steps=None):
        """Perform Gibbs sampling from the current state values.
        Args:
          steps (int) : number of steps of sampling (passes through all variables) to perform
        Return:
          D (list[tuple]) : data array containing the state(s) of the Markov chain after sampling
        """
        if steps is None: steps = self.steps
        for i in range(steps):
          for b in self.blocks:
            for c,s in enumerate(self.states):  # for each "chain"
              #for x in b: s.pop(x)
              if self.model.isLog:
                p = reduce(lambda f,g:f+g, (f.condition({v:s[v] for v in f.vars-b}) for f in self.model.factorsWithAny(b)) )
                #p = reduce(lambda f,g:f+g, (f.condition(s) for f in self.model.factorsWithAny(b)) )
                p -= p.lse();
                p.expIP()
              else:
                p = reduce(lambda f,g:f*g, (f.condition({v:s[v] for v in f.vars-b}) for f in self.model.factorsWithAny(b)) )
                #p = reduce(lambda f,g:f*g, (f.condition(s) for f in self.model.factorsWithAny(b)) )
                p /= p.sum()
              xb = p.sample()
              for x,sx in zip(b,xb): self.states[c][x]=sx
        return [tuple(s) for s in self.states]               # copy/freeze before return
        #return [{i:x[i] for i in x} for x in self.states]   # copy before return




#
# TODO: Change Metrop as with GibbsSampler
#


def Metropolis( model, query, proposal, state=None, stopSamples=1, stopTime = inf ):
    """Metropolis-Hastings sampling for discrete graphical model "model" with query object "query"
       proposal : function object: newstate,logratio = proposal(oldstate) 
                  propose random transition "old -> new" and report log q(old->new)/q(new->old)
    """
    state = state if state is not None else [np.random.randint(Xi.states) for Xi in model.X]
    j = 0; stopTime += time();
    while (j < stopSamples) and (time() < stopTime):
        stateNew, logq_ratio = proposal(state)
        # TODO: can be more efficient to only evaluate *change* in state (use logValue(state, subset))
        logf_ratio = model.logValue(stateNew) - model.logValue(state)
        if np.log(np.random.rand(1)) < logf_ratio + logq_ratio: state = stateNew
        query.update(state,1.0)
        j += 1
    # TODO: some way to track accept/reject ratio easily?
    return query   # TODO: way to return state also, for continuation?  Make object?




#  use functions?  "marginals(model)(x,w) => update list of marginals in model"
#      "mapconfig(F)(x,w) => eval F(x) & update if larger"
#      "expectation(G)(x,w) => update expectation of G(x)   (needs step #?)
#      "stateSequence()(x,w) => append x to state sequence list
#      "expectSequence(G)(x,w) => append G(x) to sequence


# thoughts...
#  query only single variable marginals?
#  query all cliques in the model?
#  query one or some cliques in the model?
#  query expectation of some function?
#  query "trace" of a function (possibly identity => state sequence)
#  query generic operator, e.g., keep argmax_xi f(xi) ?
#     (could use for expectations with callable object?)


### Forward / rejection sampling  (for BNs only?)
###    Verify GM is a BN; sample x; reject if not Xe; update query

### Likelihood weighting (BNs only?  Or any with topo order conditionals?)
###    Verify GM is a BN; sample Xh & eval Xe

### Importance sampling (generic form?)
###   f(x), q(x) draw and eval; queries
###   


#
# TODO: Convert to sampler object as GibbsSampler()
#

def ImportanceSampling(model, query, proposal, stopSamples=1, stopTime = inf):
    j = 0; stopTime += time();
    while (j < stopSamples) and (time() < stopTime):
        xi,qi = proposal()
        fi = model.logValue(xi)
        query.update(xi,np.exp(fi-qi))   # exp? do weights in log form?
        j += 1
    return query

def AnnealedImportanceSampling(model, query, base, T=10, K=5): #stopSamples=1, stopTime=inf):
    #j = 0; stopTime += time()
    model_eps = model.copy(); model_eps.makePositive();
    if True: #while (j < stopSamples) and (time() < stopTime):
        xi,ln_wi = bnSample(base, bnOrder(base)) # TODO: how? bnSample?  wi = 1?
        current = base
        xi = list(xi)
        for t in np.linspace(1./T,1,T):
            annealed = GraphModel( [f**(1.-t) for f in base.factors]+[f**t for f in model_eps.factors] )
            ln_wi += (annealed.logValue(xi) - current.logValue(xi))
            gs = GibbsSampler(annealed,chains=1,init_states=[xi])
            xi = list(gs.sample(K)[0])
            current = annealed
            #xi = GibbsSamplingBlock(annealed, QuerySamples(1,1), state=xi, stopSamples=stopSamples)[0]
        ln_wi += (model_eps.logValue(xi) - model.logValue(xi))
        #query.update(xi,np.exp(ln_wi))
    return tuple(xi),ln_wi
    #return query

### Annealed IS:
###   provide base proposal p0, target f, # temp, positivity eps, Gibbs method (cliques, # steps)
###   sample x ~ p0, define p+ = f/Zhat+eps, pt = (p0)^(1-t)*(p+)^t, t=1/#t; w = pt(x)/f 
###   for t=2/#t...1: MCMC(pt,x); w*=pt+1/pt;
###   w*=f/p+
###      ** if p0 factors like p+, we can use local ops for gibbs. what if not? (enum?) (pass GModel vs pass eval f'n)

 
