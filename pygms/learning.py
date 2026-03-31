################################################################################
## IMPORTS #####################################################################
################################################################################


import numpy as np

from numpy import asarray as arr
from numpy import atleast_2d as twod

from itertools import combinations as combs
from itertools import permutations as perms


#########################################
# Simple parameter estimation methods?
#########################################
# BN : empirical; empirical_update (or other incremental fitting?)
#      (for updating function: map factor to current step size? argument? ?)
#
# IPF : again, full step vs partial step?  order of visitation?
#    Inference method? Preselect or pass in (VE, JT, etc)?
#
# BP fixed point (IPF with no inference? Set unary, then binary by phat/unaries)
#
# Direct gradient optimization?  SGD or AutoGrad?
#   Exact LL?    (regularize? other?)
#   PseudoLL
#   Other surrogates (WMB, etc)
#
# fit e.g. CD into these frameworks easily?
# preserve BN parameterization if desired? (project after update?)
#
# fit stochastic em or hard em into frameworks? (infer data; incremental step; repeat)
#
# helper / related methods that take expectations over cliques instead of data?
#   * use for actual EM: accumulate expectations via inference then update
#



#########################################
# Score functions & complexity penalties
#########################################

def bic(p,m): 
    """Bayesian Information complexity penalty (Schwarz et al 1978)
      For *average* log-likelihood; p = # parameters, m = # data  (so, ~ p*log(m)/m )"""
    return p*np.log(m)/2/m
def aic(p,m): return None     # Aikike info criterion
def qnml(args): return None   # Quotient normalized maximum likelihood (Silander et al 2018)

def bde(counts,i,alpha=1.):
    """Bayesian Dirichlet equivalent score function (Heckerman et al. 1995)"""
    return _bdgeneral(counts,i,gm.Factor(counts.vars,alpha/counts.numel))

def _bdgeneral(counts,i,alphas):
    """Internal helper function for Bayesian Dirichlet equivalent score f'n"""
    from scipy.special import logggamma  # could use math.lgamma in Python 3.2+
    pseudocounts = counts + alphas
    result = gm.Factor(pseudocounts.vars, 0.)
    result.table = loggamma( pseudocounts.table ) - loggamma( alphas.table )
    result = result.sum(i)
    result.table += loggamma( alphas.sum(i).table ) - loggamma( pseudocounts.sum(i) )
    return float( result.sum() )




#########################################
# Bayes Net structure learning
#########################################

### Brute force search (inefficient)

def bnStructure(Data, **kwargs):
    """Brute force enumeration over all orderings (inefficient!)"""
    m,n = Data.shape
    BIC = -np.inf;
    for p in perms(range(n)):
        Pp, BICp = bnStructGivenOrder( Data, order=p , **kwargs)   # permute data & find parents
        if BICp > BIC: 
            parents,BIC = Pp,BICp                       # save the best structure so far
    return parents,BIC


def bnStructGivenOrder( Data, maxpa, order=None, alpha=1e-6, penalty=bic, saved_scores=None, verbose=False):
    """Find best Bayes Net structure given a topological variable order. 
      maxpa (int) : maximum number of parents to consider
      order (list[int]) : topological order to enforce in the BN
      alpha (float) : regularization for empirical probability estimates
      penalty (function; default bic) : decomposable complexity penalty to include in score
      saved_scores (dict[(x,xpa)]) : cached scores to avoid recomputation
      verbose (bool) : display detailed search information
    """
    m,n = Data.shape
    sz = Data.max(axis=0)+1   # largest state value in dataset
    X = [gm.Var(i, s) for i,s in enumerate(sz)]
    if order is None: order = np.arange(n,dtype=int)   #  default order
    else: order = np.array(order, dtype=int)
    pri = np.zeros((n,)); pri[np.array(order)] = range(n);
    parents = [[] for x in X]
    BIC = 0
    for i in range(n):
        bestBIC = -np.inf
        for npa in range(maxpa+1):
            for ppa in combs(range(i),npa):
                pa = tuple(sorted(order[list(ppa)]))  # apply topological order to find parent set
                oi = order[i]
                if saved_scores is not None and (oi,pa) in saved_scores: 
                    BICp = saved_scores[(oi,pa)]
                else:
                    # Compute the empirical joint & conditional over the clique:
                    Pi = gm.empirical([[X[oi]]+[X[p] for p in pa]], Data)[0] + alpha
                    Pi /= Pi.sum()
                    Ci = Pi / Pi.sum([oi])
                    nParam = Pi.numel() * (1-1./X[oi].states)
                    # Compute average log-likelihood over the data, minus BIC penalty:
                    BICp = (Pi*Ci.log()).sum() - penalty(nParam,m)
                    if saved_scores is not None: saved_scores[(oi,pa)] = BICp
                    if verbose: print(f"{i}: {oi}|{pa} : {BICp}")
                # If this is the best score so far, save it:
                if BICp > bestBIC: parents[oi],bestBIC = pa,BICp
                # TODO: should save best Ci to a factor list; or recompute when done?
        BIC += bestBIC   # sum up BIC scores for total model BIC score
        if verbose: print(f'===  {BIC}')
    return parents,BIC


""" Demo brute force learning

import numpy as np
import networkx as nx

np.random.seed(0)
Data = (np.random.rand(30,4)>.5).astype(int);
m,n = Data.shape
saved_scores = {} #None
#bnStructGivenOrder(Data, maxpa=3, penalty=lambda n,m:0)
paSets,score = bnStructure(Data, maxpa=2, penalty=bic, saved_scores=saved_scores, verbose=False) #lambda n,m: 0)

nxG = nx.DiGraph(); nxG.add_nodes_from(range(n))
for i,p in enumerate(paSets): nxG.add_edges_from([(pj,i) for pj in p])

nx.draw(nxG, node_color='w', edgecolors='k', labels={i:str(i) for i in range(n)})
print("Best score ",score)

"""

##############################################
# TODO: algorithm completions
###############################################

def bnHillClimbing(*kwargs):
    """Simple hill-climbing local search for Bayes net structure learning (Heckerman et al., 1995) """
    raise NotImplementedError("Hill climbing not yet implemented")
    # TODO: include stochastic local search?  pocket? annealed?

def bnChowLiu(*kwargs):
    pass

def 



""" Integer Linear Program method, via CVX solver


m,n = Data.shape
sz = Data.max(axis=0)+1   # largest state value in dataset
X = [gm.Var(i, s) for i,s in enumerate(sz)]
alpha = 1e-6
penalty = bic
penalty = lambda n,m: 1.*bic(n,m)


k = 2
subs = [list(combs(range(n-1),i)) for i in range(k+1)]
allParents = sum(subs, start=[])
revParents = [[i for i,p in enumerate(allParents) if j in p] for j in range(n)]


cols = len(allParents)     # each of n nodes can have any of Ch(n-1,k) parent sets
S = -np.ones((n,cols))     # scores for each possible parent set
A = np.zeros((n, n*cols))  # each row of A is a constraint on the solution vector, Ax=b
b = np.zeros(n)

# Construct initial scores and unique parent set constraints
for i in range(n):
    v = np.array(list(range(i))+list(range(i+1,n)))  # possible parent node ids
    for k,pa in enumerate(allParents):
        Pi = gm.empirical([[X[i]]+[X[p + (1 if p>=i else 0)] for p in pa]], Data)[0] + alpha
        Pi /= Pi.sum()
        Ci = Pi / Pi.sum([i])
        nParam = Pi.numel() * (1-1./X[i].states)
        # Compute average log-likelihood over the data, minus BIC penalty:
        S[i,k] = (Pi*Ci.log()).sum() - penalty(nParam,m)

    R = np.zeros((n,cols)); R[i,:] = 1;      # Initial constraint: exactly one parent set per node
    A[i,:] = R.reshape(-1); b[i] = 1;
S = S.reshape(-1)



B,I = set(range(len(S))), set()

G_list = [ list(np.ones(len(S))) ]   # trivial sum bound (initialization)
done = False

while True:
    if (len(G_list) > 200):
        print("Too many constrints?"); break;

    G = matrix(G_list).T; h = matrix([sum(g)-.5 for g in G_list]); # identified cycles:
    #print(G,h)

    status,xo = ilp(-matrix(S),G,h,matrix(A),matrix(b),I,B)
    #print('status:',status);
    x = np.array(xo).reshape(n,cols)
    #print(x)
    paSets = [[p if p<i else p+1 for p in allParents[np.argmax(x[i])]] for i in range(n)]
    score = matrix(S).T*xo

    nxG = networkx.DiGraph(); nxG.add_nodes_from(range(n))
    for i,p in enumerate(paSets): nxG.add_edges_from([(pj,i) for pj in p])
    #networkx.draw(nxG, labels={i:str(i) for i in range(n)})
    #plt.show()
    print("Current score", score[0])

    try:
        cycle = networkx.find_cycle(nxG, orientation='original')

        block_children = [j for i,j,d in cycle]
        for jj in range(n):
            if jj not in block_children: x[jj,:] *= 0
        G_list.append( list(x.reshape(-1)) )
    except:
        break  # no cycle => exit loop

networkx.draw(nxG, labels={i:str(i) for i in range(n)})
plt.show()
print("Score", score[0])

"""


