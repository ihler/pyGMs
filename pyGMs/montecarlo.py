"""
montecarlo.py

Defines several Monte Carlo and MCMC routines for approximate inference in graphical models

Version 0.1.1 (2022-04-06)
(c) 2015 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import numpy as np
#import time as time
from time import time
from sortedcontainers import SortedSet;
from builtins import range

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

# Basic query interface:
#   Q = Query( f )                   # to compute expectation of f(x): E_p[f] 
#   Q = Query( [f1, f2, ...] )       # to compute E_p[fi] for each fi in the list
#   Q = QueryMarginals( factorlist ) # to compute p(x_a) for each factor f_a(x_a) in the list
#
# A query is an object Q=Query(..) with:
#   Q() or Q[i]   : return the current estimate, or ith estimate from a list
#   Q.update(x,w) : update the expectations' estimates by observing state x with weight w
#   Q.wvar()      : return the empirical variance of the weights encountered (optional)


class Query(object):
    """Defines a Monte Carlo "query" object, for estimation of various quantities from sequences of (weighted) states.
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
    """
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

###################################################### 
class QueryMarginals(Query):
    """Specialized Monte Carlo "query" object for marginal probabilities of factors
       Q = QueryMarginals( factorlist ) # estimate marginal p(x_a) for each factor f_a(x_a) in list
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





 
