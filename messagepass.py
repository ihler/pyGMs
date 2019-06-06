import numpy as np
import time as time
from sortedcontainers import SortedSet;

from pyGM.factor import *
from pyGM.graphmodel import *
from builtins import range

inf = float('inf')




# Basic implementation -- flooding schedule f->v, v->f etc.
#
#
#

def LBP(model, maxIter=100, verbose=False):
    beliefs_F = [ f/f.sum() for f in model.factors ]       # copies & normalizes each f
    beliefs_V = [ Factor([v],1.0/v.states) for v in model.X ] # variable beliefs
    msg = {}
    for f in model.factors:
        for v in f.vars:
            msg[v,f] = Factor([v],1.0)  # init msg[i->alpha]
            msg[f,v] = Factor([v],1.0)  # and  msg[alpha->i]
    
    for t in range(1,maxIter+1):               # for each iteration:
        # Update beliefs and outgoing messages for each factor:
        for a,f in enumerate(model.factors):
            beliefs_F[a] = f.copy()                 # find f * incoming msgs & normalize
            for v in f.vars: beliefs_F[a] *= msg[v,f]
            beliefs_F[a] /= beliefs_F[a].sum()      #   divide by i->f & sum out all but Xi 
            for v in f.vars: msg[f,v] = beliefs_F[a].marginal([v])/msg[v,f]
        # Update beliefs and outgoing messages for each variable:
        for i,v in enumerate(model.X):
            beliefs_V[i] = Factor([v],1.0)       # find product of incoming msgs & normalize
            for f in model.factorsWith(v): beliefs_V[i] *= msg[f,v]
            beliefs_V[i] /= beliefs_V[i].sum()      #   divide by f->i to get msg i->f
            for f in model.factorsWith(v): msg[v,f] = beliefs_V[i]/msg[f,v]
            
        #for f in model.factors:                    # print msgs and beliefs for debugging
        #    for v in f.vars:
        #        print v,"->",f,":",msg[X[v],f].table
        #        print f,"->",v,":",msg[f,X[v]].table
        #for b in beliefs_F: print b, b.table
        #for b in beliefs_V: print b, b.table

        # Compute estimate of the log partition function:
        # E_b [ log f ] + H_Bethe(b) = \sum_f E_bf[log f] + \sum_f H(bf) + \sum (1-di) H(bi)
        lnZ = sum([(1-len(model.factorsWith(v)))*beliefs_V[v].entropy() for v in model.X])
        for a,f in enumerate(model.factors):
            lnZ += (beliefs_F[a] * f.log()).sum()
            lnZ += beliefs_F[a].entropy()
        if verbose: print("Iter "+str(t)+": "+str(lnZ))
    return lnZ,beliefs_V


#@do_profile(follow=[get_number])
def NMF(model, maxIter=100, beliefs=None, verbose=False):
    """Simple naive mean field lower bound on log(Z).  Returns lnZ,[bel(Xi) for Xi in X]"""
    if beliefs is None: beliefs = [Factor([Xi],1.0/Xi.states) for Xi in model.X]
        
    lnZ = sum([beliefs[Xi].entropy() for Xi in model.X])
    for f in model.factors:
        m = f.log()
        for v in f.vars: m *= beliefs[v]
        lnZ += m.sum()
    if verbose: print("Iter 0: "+str(lnZ))

    for t in range(1,maxIter+1):               # for each iteration:
        # Update all the beliefs via coordinate ascent:
        for Xi in model.X:                      # for each variable, 
            bNew = 0.0                          # compute E[ log f ] as a function of Xi:
            for f in model.factorsWith(Xi,copy=False): #   for each factor f_a, compute:
                m = f.log()                     #   E[log f_a] = \sum \log f_a \prod b_v
                for v in f.vars - [Xi]: m *= beliefs[v]
                bNew += m.marginal([Xi])        #   sum them up to get E[log f]
                bNew -= bNew.max()              # (numerical issues)
            bNew = bNew.exp()
            bNew /= bNew.sum()                  # set b(Xi) = exp( E[log f] ) / Z
            beliefs[Xi] = bNew
        #
        # Compute the lower bound on the partition function:
        # E_b [ log f ] + H(b) = \sum_a E[log f_a] + \sum_i H(b_i) for independent beliefs
        lnZ = sum([beliefs[Xi].entropy() for Xi in model.X])
        for f in model.factors:
            m = f.log()
            for v in f.vars: m *= beliefs[v]
            lnZ += m.sum()
        if verbose: print("Iter "+str(t)+": "+str(lnZ))
    return lnZ,beliefs




################ DECOMPOSITION METHODS #############################################
#@do_profile(follow=[get_number])
def DualDecomposition(model, maxIter=100, verbose=False):
    """ ub,lb,xhat = DualDecomposition( model [,maxiter,verbose] )
        Compute a decomposition-based upper bound & estimate of the MAP of a graphical model"""
    lnF       = sum(  np.log(f.max()) for f in model.factors )
    lnX, xhat = -np.inf, np.zeros( (len(model.X),), dtype=int) 
    lnR, rhat = -np.inf, np.zeros( (len(model.X),), dtype=int) 
    if verbose: print("Iter 0: "+str(lnF))
            
    for t in range(1,maxIter+1):               # for each iteration:
        # Update each variable in turn:
        for Xi in model.X:                      # for each variable, 
            flist = model.factorsWith(Xi, copy=False)
            gamma = [f.maxmarginal([Xi]) for f in flist]
            avg   = np.prod(gamma)**(1.0/len(gamma))
            for f,g in zip(flist,gamma): f *= avg/(g+1e-300)   # !!! numerical issues... 
            xhat[Xi] = avg.argmax()[0]          # guess a state for Xi
        #
        # Compute the upper bound on the maximum and the value of our current guess
        lnF = sum( np.log(f.max()) for f in model.factors )
        lnX = model.logValue( xhat )
        if lnR < lnX:  lnR = lnX; rhat[:]=xhat;
        if verbose: print("Iter "+str(t)+": "+str(lnF)+" > "+str(lnX))
        if (lnF == lnX): break
    return lnF,lnR,rhat


  

def WeightedDD( factors, weights, elimOrder, direction=1.0, maxIter=100, verbose=False, stop_tol=0.0 ):
    step_inner = 5;
    thetas = [f.log() for f in factors]
    weights = { th:wt for th,wt in zip(thetas,weights) }
    logmodel = GraphModel(thetas, copy=False)

    def wt_elim(f,w,pri):
      elim_ord = np.argsort( [pri[x] for x in f.vars] )
      tmp = f
      for i in elim_ord: tmp = tmp.lsePower([f.v[i]],1.0/w[i])
      return tmp

    def calc_bound( thetas, weights, pri):
      return sum([wt_elim(th,wt,pri) for th,wt in zip(thetas,weights)])

    def calc_deriv(th,w,pri,match,Xi=None):
      elim_ord = np.argsort( [pri[x] for x in th.vars] )
      lnZ0 = th.copy()
      lnmu = 0.0 * lnZ0
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

    def update_weights(weights,idx,dW,stepW):   # TODO only works for positive weights
      wtot = 0.0
      for j,wt,dw in zip(idx,weights,dW): wt[j] *= np.exp( - stepW * wt[j] * dw ); wtot += wt[j];
      for j,wt,dw in zip(idx,weights,dW): wt[j] /= wtot;

    def armijo(thetas,weights,pri,Xi,steps,threshold=1e-4,direction=+1, optTol=1e-8,progTol=1e-8):
      import copy
      f0,f1 = None, calc_bound(thetas,weights,pri)                       # init prev, current objective values
      match = reduce(lambda a,b: a&b, [th.vars for th in thetas], thetas[0].vars)
      idx = [th.v.index(Xi) for th in thetas] if Xi is not None else []  # find location of Xi in var/weight vectors
      newweights = copy.deepcopy(weights) if Xi is not None else weights # copy weights if updated
      for s in range(steps):
        # compute gradients dPhi/dTheta, dPhi/dW  (wrt parameters, weights):
        dT,dW = zip(*[calc_deriv(th,wt,pri,match,Xi) for th,wt in zip(thetas,weights)])
        dT,dW = list(dT),list(dW)
        for dt in dT[1:]: dt -= dT[0]; dt *= -1;
        dT[0] = -sum(dT[1:])
        if Xi is not None: 
          Hbar = sum([wt[j]*dw for j,dw,wt in zip(idx,dW,weights)])
          for j in range(len(dW)): dW[j] -= Hbar
        # Compute gradient norms:
        L0,L1,L2 = zip(*[ (d.max(),d.sum(),(d*d).sum()) for dt in dT for d in [dt.abs()]])
        L0,L1,L2 = max(L0),sum(L1)+1e-300,sum(L2)
        L0,L1,L2 = max(L0,max(abs(dw) for dw in dW)), L1+sum(abs(dw) for dw in dW), L2+sum(dw*dw for dw in dW)
        if L0 < optTol: return                              # if < optTol => local optimum  

        step = min(1.0, 1.0/L1) if f0 is None else min(1.0, direction*(f0-f1)/L1)
        step = step if step > 0 else 1.0
        f0 = f1;                                            # update "old" objective value
        for dt in dT: dt *= direction*step;                 # premultiply step size into dT
        for j in range(10):
          newthetas = [th+dt for th,dt in zip(thetas,dT)]   # step already pre-multiplied
          if Xi is not None: update_weights( newweights, idx, dW, step );
          f1 = calc_bound(newthetas,newweights,pri)
          #print "  ",f0," => ",f1, "  (",f0-f1,' ~ ',stepsize*threshold*gradnorm,")"
          if (f0 - f1)*direction > step*threshold*L2:       # if armijo "enough improvement" satisfied
            for th,nth in zip(thetas,newthetas): th.t[:] = nth.t   # rewrite tables
            for j,wt,w2 in zip(idx,weights,newweights): wt[j] = w2[j];
            break;
          else:                                             # ow, back off 
            step *= 0.5; 
            if step*L0 < progTol: return                    # if < progTol => no progress possible
            for dt in dT: dt *= 0.5

    elimOrder = np.asarray(elimOrder);
    pri = np.zeros((elimOrder.max()+1,))
    pri[elimOrder] = np.arange(len(elimOrder))
    #
    lnZw = calc_bound(thetas,[weights[th] for th in thetas],pri)
    start_time = time.time()
    if verbose: print("Iter 0: "+str(lnZw))
    for t in range(1,maxIter+1):               # for each iteration:
        # Update each variable in turn:
        for Xi in logmodel.X:                      # for each variable, 
            theta_i = logmodel.factorsWith(Xi) 
            if len(theta_i) <= 1: continue;
            weight_i = [weights[th] for th in theta_i]
            armijo(theta_i,weight_i,pri,Xi, 5, 0.01, direction)
        #
        # Compute the upper bound on the maximum and the value of our current guess
        prev, lnZw = lnZw, calc_bound(thetas,[weights[th] for th in thetas],pri)
        if verbose: print("[{}] Iter {} : {}".format(time.time()-start_time,t,lnZw));
        if (prev - lnZw)*direction < stop_tol: break
    return lnZw, thetas



