"""
Weighted mini-bucket elimination for graphical models
Computes upper or lower bounds on the partition function or MAP/MPE configurations, depending on weights
Supports incremental construction
Supports TRW-based importance sampling

class WMB:
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

from builtins import range
try:
    from itertools import izip    # OLD python versions required this
except:
    izip = zip                    # more recent versions we can just do this


reverse_enumerate = lambda l: izip(range(len(l)-1, -1, -1), reversed(l))

class WMB(object):
    '''Class implementing weighted mini-bucket elimination inference.
    See documentation and example notebooks for usage.
    '''

    # Internal object / structure for representing a mini-bucket
    class Node:
        """Internal container object for mini-bucket nodes"""
        def __init__(self):
            self.clique = VarSet()
            self.theta = Factor().log()
            self.weight = 1.0
            self.parent = None
            self.children = []
            self.msgFwd = Factor().log()
            self.msgBwd = Factor().log()
            self.originals = []
        def __repr__(self):
            return "{}^{}".format(self.clique,self.weight)
        def __str__(self):
            return "{}".format(self.clique)
        def __lt__(self, other):
            return False         # don't care about ordering nodes


    class ConstantList:
        """Internal helper object for treating scalars and lists interchangably"""
        def __init__(self, val):
            self.val = val
        def __getitem__(self, loc):
            return self.val


    def __init__(self, model, elimOrder=None, iBound=None, sBound=None, weights=1.0, attach=True, **kwargs):
        """Construct a mini-bucket object from a given graphical model.
          model : GraphModel object representing the graphical model
          elimOrder : an elimination order (list-like) or method (e.g. "minfill")
          iBound : maximum number of variables in any mini-bucket clique
          sBound : maximum number of state configurations for any mini-bucket clique
          weights: weighted elimination per bucket; see setWeights()
          attach : construct clique functions (True) or postpone until later (False)
        """
        # TODO: check if model isLog() true
        # save a reference to our model
        self.model = model
        self.X     = model.X
        self.logValue = model.logValue

        # create & process elimination ordering of the model:
        if elimOrder is None: elimOrder = 'wtminfill'
        if type(elimOrder) is str:   # auto elim order: check that weights is string or float
            if not type(weights) in {float, str}:
              raise ValueError("Must specify elimination order or use all-equal weights (float or string)"); 
            elimOrder = eliminationOrder(self.model, orderMethod=elimOrder)[0];  
        self.elimOrder = elimOrder
        self.priority = [-1 for i in range(model.nvar)]  # build priority of each var
        for i,x in enumerate(elimOrder): self.priority[x] = i

        # now build the mini-bucket data structure
        self.buckets = [ [] for x in range(model.nvar) ]  # bucket for each var: list of minibuckets 
        self.matches = [ [] for x in range(model.nvar) ]  # matching sets for each bucket
        self.weights = WMB.ConstantList(1.0)              # default to sum+ during construction
        for f in model.factors:
            if len(f.vars)==0: continue;      #TODO: should add anyway (somewhere)
            n = self.addClique(f.vars)
            if attach: n.theta += f if model.isLog else f.log()  # include log f(x) in node's log-factor
            n.originals.append(f)              # append (pointer to) original f for later reference
        if iBound is not None or sBound is not None:      # perform merging if criteria given at init
            self.merge(self.scoreByScope(ibound=iBound,sbound=sBound))
        self.setWeights(weights)                          # and set the weights of the buckets
 
    def setWeights(self,weights):
        """Set the weights of the inference problem.
        weights = 'max+' or 0.0  => upper bound the MAP configuration
                  'sum+' or 1.0  => upper bound the partition function
                  'sum-' or -1.0 => lower bound the partition function
        For more general bounds, weights = list of floats (one per variable)
        """
        if type(weights) is str:
            if weights == 'sum+': weights =  1.0;
            elif weights=='sum-': weights = -1.0;
            elif weights=='max+': weights = 1e-8;
            else: raise ValueError("Unknown weight / task type; must be max+, sum+, sum-, or float / float list")
        if type(weights) is float: weights = WMB.ConstantList(weights)
        self.weights = weights

        for i,xi in enumerate(self.elimOrder):     # (TODO?) set mini-bucket weights uniformly
            ni = len(self.buckets[i])
            for j in range(ni):               # set uniformly
                self.buckets[i][j].weight = self.weights[xi]/ni
            if self.weights[xi] < 0 and ni > 0:          # uniform for lower bound:
                self.buckets[i][0].weight = 1.0 - self.weights[xi]*(ni-1)/ni



 
    def __nodeID(self,node):
        """Helper function: get identifier (bucket & location) of a given node""" 
        if not isinstance(node, WMB.Node): return None,None
        i = min([self.priority[x] for x in node.clique])  # get bucket
        j = self.buckets[i].index(node)
        return i,j
   
    def __repr__(self):     
        """Detailed representation of the mini-bucket structure as a text string"""
        to_return = ""
        for i,b in enumerate(self.buckets):
            to_return += "{:03d}: ".format(int(self.elimOrder[i]))
            for j,mb in enumerate(b):
                to_return += "{!s}^{:.2f} => {}; ".format(mb,mb.weight, self.__nodeID(mb.parent))
            to_return += "\n"
        return to_return

    def draw(self):
        """Draw mini-bucket tree in networkx.  Limited usefulness without good positioning.
        Returns the networkx DiGraph of the mini-bucket tree.
        """
        import networkx as nx
        pos,labels = {},{}
        roots = []
        G = nx.DiGraph()
        for i,b in enumerate(self.buckets):
            for j,mb in enumerate(b):
                G.add_node(str(mb))
                pos[str(mb)] = (j,-i)
                labels[str(mb)] = str(mb)
        for i,b in enumerate(self.buckets):
            for j,mb in enumerate(b):
                if mb.parent is not None: G.add_edge(str(mb),str(mb.parent))
                else: roots.append(mb)
        # Revise x-positions to respect descendent positions
        def _revise(self, root, w=1., x=0.5, pos=None, par=None):
            if type(root) is list: children = root
            else:
                pos[str(root)] = (np.round(x,4), pos[str(root)][1]) 
                children = root.children
            if len(children):
                dx = w/(len(children))
                x_ = x - w/2 - dx/2
                for ch in children:
                    x_ += dx
                    pos = _revise(self,ch, w=dx, x=x_, pos=pos, par=root)
            return pos
        pos = _revise(self,roots,pos=pos)

        # Now revise x-positions more uniformly
        xvals = np.unique([pos[p][0] for p in pos])
        xmap = {x:i*1./len(xvals) for i,x in enumerate(xvals)}
        for p in pos: pos[p]=(xmap[pos[p][0]],pos[p][1])

        nx.draw(G, pos=pos, labels=labels)
        return G



    def addClique(self,vars):
        """Add a clique with scope "vars", fixing up structure to be a valid MB tree"""
        vs = VarSet(vars)
        corder = np.argsort( [self.priority[x] for x in vars]  ) # get order in which eliminated
        corder = [vars[c] for c in corder]
        added = []
        found = False
        for x in corder:
            if found: break
            #print("bucket ",x)
            b = self.buckets[self.priority[x]]
            to_remove = []
            for mb in b:
                #print("  check ",mb)
                if mb.clique < vs:
                   to_remove.append(mb)
                if mb.clique >= vs:                # if we found a minibucket we can just join, do:
                    if len(added) > 0:             # if we've added nodes, connect them as descendants
                        mb.children.append( added[-1] )  # of the found node, and found node as parent of last
                        added[-1].parent = mb          
                    found = True                   # now, we don't need to keep generating parents
                    #print("    Found!")
                    added.append(mb)               # not really added, but the end of the added chain
                    break
            # if we didn't find any mini-buckets we can join, we need to add one:
            if not found:                          # 
                n = WMB.Node()
                n.clique = VarSet(vs)
                n.weight = -1e-3 if self.weights[x] < 0 else 1e-3;   # TODO: small non-zero weights
                #print("adding ",n," to ",self.priority[x])
                b.append(n)
                if len(added) > 0:                 #   then, last added node is the child of this one
                    n.children.append(added[-1])
                    added[-1].parent = n
                added.append(n)                    #   put in added list 
                vs -= [x]                          #   next bucket is what's left after x is eliminated
                for mb in to_remove:
                    for c in mb.children: c.parent = n  # change children to point to new node, and
                    n.children.extend(mb.children) #   merge with current child list
                    n.weight += mb.weight          # join weights into new node
                    if mb.parent is not None:      # if mb has a parent, shift factors around to preserve bound
                        mb.theta -= mb.msgFwd
                        mb.parent.theta += mb.msgFwd
                        mb.parent.children.remove(mb)
                    n.theta += mb.theta                # join log-factors into new node
                    n.originals.extend(mb.originals)   # move original factor pointers to new node
                    b.remove(mb)
                #n.theta += Factor(n.clique,0.0);  # need to do this at some point to ensure correct elim
                # TODO: fix up match structure?
        # done adding all required cliques; return 1st 
        return added[0]


    def detachFactors(self):
        """Remove factor tables from their associated cliques; speeds up scope-based merging"""
        for b in self.buckets:
            for mb in b:
                mb.theta = Factor([],0.)

    def attachFactors(self):
        """Re-attach factor tables to their associated cliques for evaluation"""
        for b in self.buckets:
            for mb in b:
                mb.theta = Factor([],0.)    # TODO: need to ensure Xb is there in case no factor includes it?
                for f in mb.originals: 
                    mb.theta += f if self.model.isLog else f.log()  # TODO: check if already in log form?

    def memory(self, bucket=None, use_backward=True):
        """Estimate the total memory (in MB) required for this mini-bucket approximation"""
        mem = 0.
        use_buckets = self.buckets if bucket is None else [self.buckets[bucket]]
        for b in use_buckets:
            for mb in b:
                mem += mb.clique.nrStatesDouble() * mb.theta.table.itemsize
                # TODO: add forward & backward message costs here also
        return mem / 1024. / 1024.

    # TODO: convert to external function?  pass variable in; check if refinement of another?
    def scoreByScope(self, ibound=None, sbound=None):
      """Returns a scope-based scoring function for use in merge()"""
      def score(m1,m2):
        jt = m1.clique | m2.clique
        if ibound is not None and len(jt) > ibound: return -1
        if sbound is not None and jt.nrStates() > sbound: return -1
        # TODO: also disallow if not consistent with some specified scope sets?
        mx,mn = max([len(m1.clique),len(m2.clique)]), min([len(m1.clique),len(m2.clique)])
        #return 1.0/(float(mx)+float(mn)/mx)   # prefers merging smaller cliques first
        return (float(mx)+float(mn)/(mx+1))    # prefer merging largest cliques
      # return the scoring function 
      return score


    # score = len(max)+len(min)/len(max) if union < iBound else -1 for scope
    def merge(self, score):
        """merge(score_function) : perform score-based merging of mini-buckets
        Merging cliques of the mini-bucket tree typically improves accuracy at a computational cost.
        score : a function, score(m1,m2), that evaluates a pair of mini-bucket cliques,
                determines if they can be merged, and prioritizes them.
                Higher value pairs are merged first; a negative return value indicates an invalid merge.
                See WMB.scoreByScope() for an example.
        """
        from heapq import heappush,heappop
        try:
          from itertools import count
          tiebreak = count().__next__        # need tiebreaker value for priority queue (?)
        except:
          tiebreak = lambda: 0
        for b in self.buckets:
            priority = [] 
            lookup = {}
            # see not-too-efficient-looking remove_task https://docs.python.org/2/library/heapq.html
            # build scores:
            for i,m1 in enumerate(b):
                for m2 in b[i+1:]:
                    s = score(m1,m2)
                    if s >= 0.0:                       # if the merge is allowed,
                        entry = [-s,tiebreak(),m1,m2]  # add it to the priority heap
                        lookup[(m1,m2)] = entry
                        heappush(priority, entry)
            while len(priority):
                entry = heappop(priority)              # find the best pair to merge:
                s,_,m1,m2 = entry[0],entry[1],entry[2],entry[3]
                #s,m1,m2 = priority.pop()
                if m1 is None or m2 is None: continue
                if m1 not in b or m2 not in b: continue   ## check for removed minibuckets?
                #print("Merging ",m1,"+",m2)
                for m in b:                            # mark anything involving m1 or m2 as now invalid
                    for ma,mb in [(m1,m),(m,m1),(m2,m),(m,m2)]:
                        s = -lookup.get( (ma,mb), [1,None,None] )[0]
                        if s >= 0.0: 
                            entry = lookup.pop( (ma,mb) )
                            entry[2],entry[3] = None,None  # mark as "removed"
                            #priority.remove( [s,ma,mb] )
                m12 = self.addClique( m1.clique | m2.clique )
                # what if others removed?  (bad check above?)
                #print(b)
                for m in b:                            # now check possible merges with new m1+m2
                    if m is m12: continue
                    s = score(m12,m)
                    if s >= 0.0:                       # if the merge is allowed, add it to queue
                        entry = [-s,tiebreak(),m12,m] 
                        lookup[ (m12,m) ] = entry
                        heappush(priority,entry)
        return None


    ### Removed; equivalent to wmb.merge( wmb.scoreByScope(iB, sB) )
    #def mergeScope(iBound=0, sBound=0): 
    #    for Bi in self.buckets:
    #        # TODO: sort bucket by size (or store sorted)
    #        for mb in Bi:
    #            # TODO: merge into largest clique that can fit
    #            pass

    #@profile
    def msgForward(self, stepTheta=0.5, stepWeights=0.1):
        """Compute a forward pass through all nodes and return the resulting bound
          stepTheta, stepWeights : step size of reparameterization updates (theta) and weights.
            Ex: stepTheta = 1., stepWeights = 0.  => "moment matching" downward pass
                stepTheta = 0., stepWeights = 0.  => Basic mini-bucket forward computation
        Returns the current bound on the mini-bucket objective (lnZ or lnF, depending on weight settings)
        """
        bound = 0.0
        for i,b in enumerate(self.buckets):
            X = self.model.vars[ self.elimOrder[i] ]
            nNodes = len(b)
            beliefs = [ None for mb in b ]
            if nNodes > 1:  # if more than one mini-bucket partition, pre-compute beliefs:
                if stepTheta or stepWeights:
                    for j,mb in enumerate(b):
                        beliefs[j] = mb.theta + mb.msgBwd
                        for c in mb.children: beliefs[j] += c.msgFwd
                        if beliefs[j].nvar < len(mb.clique): beliefs[j] += Factor(mb.clique - beliefs[j].vars,0.0);
                        beliefs[j] *= 1.0/mb.weight
                        beliefs[j] -= beliefs[j].lse()
                # Then, update theta (parameterization) on each set in "matches"
                if stepTheta:  # TODO: fix; delta / belief indexing issue
                    #for match in self.matches[i]:   # TODO: this is a bit dangerous if not kept up to date!
                    if True:                         # TODO: simple version: just match all minibuckets
                        match = b   
                        wTotal = sum([mb.weight for mb in match])
                        vAll = VarSet(match[0].clique)
                        for mb in match[1:]: vAll &= mb.clique
                        delta = [None for mb in match]
                        avg_belief = Factor().log()
                        #print "Bucket",b
                        #print match
                        #print vAll
                        for j,mb in enumerate(match):
                            delta[j] = beliefs[j].lse( mb.clique - vAll )  # TODO: belief[j] incorrect if match != b
                            #print mb.theta.table #delta[j].table
                            avg_belief += delta[j] * mb.weight / wTotal
                        #print avg_belief.table
                        #print "==="
                        for j,mb in enumerate(match):
                            delta[j] = avg_belief - delta[j]
                            beliefs[j] += delta[j] * stepTheta
                            beliefs[j] -= beliefs[j].lse()
                            mb.theta += delta[j] * mb.weight * stepTheta
                            #print mb.theta.table
                # Last, update weights if desired:
                if stepWeights:
                    isLower=(self.weights[i] == -1) # TODO: a bit difficult; needs to know weight constraints (+1/0/-1, etc.)
                    H = [0.0 for mb in b]
                    Havg = 0.0
                    totalWeight = 0.0
                    positive_node = None
                    for j,mb in enumerate(b):
                        H[j] = - (beliefs[j].exp() * (beliefs[j] - beliefs[j].lse([X]))).sum() 
                        if not isLower:
                            Havg += mb.weight * H[j]
                        elif mb.weight > 0:
                            Havg = H[j]
                            positive_node = mb
                    for j,mb in enumerate(b):
                        if not isLower:
                            mb.weight *= np.exp( -stepWeights * mb.weight * (H[j]-Havg) )
                            totalWeight += mb.weight
                        elif mb.weight < 0:
                            mb.weight *= np.exp( stepWeights * mb.weight * (H[j]-Havg) )
                            totalWeight += mb.weight
                    if not isLower:
                        for j,mb in enumerate(b): mb.weight /= totalWeight
                    else:
                        positive_node.weight = 1.0 - totalWeight

            # now, compute the forward messages:
            for j,mb in enumerate(b):
                beliefs[j] = mb.theta.copy()      # Alternative? do by re-subtracting msgBwd?
                for c in mb.children: beliefs[j] += c.msgFwd
                if beliefs[j].nvar < len(mb.clique): beliefs[j] += Factor(mb.clique - beliefs[j].vars,0.0);
                mb.msgFwd = beliefs[j].lsePower([X], 1.0/mb.weight)
                beliefs[j] = Factor().log() # clear reference & memory? 
                if mb.parent is None:       # add roots to overall bound
                    bound += mb.msgFwd  
        return float(bound)


    #@profile
    def msgBackward(self, stepTheta=0.0, stepWeights=0.0, beliefs=None):
        """Compute a backward pass through all nodes
           stepTheta, stepWeights : update bound using step sizes on clique functions & weights
           beliefs : a list of cliques (VarSets) on which to estimate marginal probabilities
           Returns a map { clique :  factor } of cliques & estimated marginals
           Raises ValueError if a requested clique is not present anywhere in the mini-bucket.
        """
        to_save = [[] for i in range(len(self.buckets))]
        if beliefs is None:
            return_beliefs = {}
        else:
            return_beliefs = { clique: None for clique in beliefs }
            # map cliques to buckets for checking
            for clique in beliefs:
                to_save[ min([self.priority[x] for x in clique]) ].append(VarSet(clique))
        for i,b in reverse_enumerate(self.buckets): 
            X = self.model.vars[ self.elimOrder[i] ]
            nNodes = len(b)
            beliefs_b = [ None for mb in b ]
            if nNodes > 1:  # if more than one mini-bucket partition, pre-compute beliefs:
                if stepTheta or stepWeights:
                    for j,mb in enumerate(b):
                        beliefs_b[j] = mb.theta + mb.msgBwd
                        for c in mb.children: beliefs_b[j] += c.msgFwd
                        beliefs_b[j] *= 1.0/mb.weight
                        beliefs_b[j] -= beliefs_b[j].lse()
                # Then, update theta (parameterization) on each set in "matches"
                if stepTheta:  
                    pass
                if stepWeights:
                    pass
            # now, compute the backward messages:
            for j,mb in enumerate(b):
                beliefs_b[j] = mb.theta + mb.msgBwd
                for c in mb.children: beliefs_b[j] += c.msgFwd
                #beliefs_b[j] -= beliefs_b[j].lse()
                #if mb.weight > 0:
                beliefs_b[j] -= beliefs_b[j].max()
                #else:
                #    beliefs_b[j] -= beliefs_b[j].min()   # invert if negative? TODO?
                beliefs_b[j] *= 1.0 / mb.weight
                for c in mb.children:
                    c.msgBwd = beliefs_b[j].lse( mb.clique - c.clique )*c.weight - c.msgFwd
                    #c.msgBwd -= c.msgBwd.max()   # TODO normalize for convenience?
                    #c.msgBwd = (beliefs_b[j]*(1.0/mb.weight)).lse( mb.clique - c.clique )*c.weight - c.msgFwd
                for c in to_save[i]:
                    if c <= mb.clique and return_beliefs[c] is None: 
                        return_beliefs[c] = beliefs_b[j].lse( mb.clique - c )
                beliefs_b[j] = Factor().log() # clear out belief
        for c,f in return_beliefs.items(): 
            if f is None: raise ValueError(f'Belief over {c} not available in WMB structure!')
            f -= f.lse()      # exponentiate and normalize beliefs before returning
            f.expIP()     
        return return_beliefs


    def reparameterize(self):
        """reparameterize() : update the current mini-bucket clique factors to incorporate the forward messages
          Useful for switching to a decomposition bound method after WMB message passing.
        """
        for i,b in enumerate(self.buckets):
          for j,mb in enumerate(b):
            if mb.parent is not None:
              mb.theta -= mb.msgFwd
              mb.parent.theta += mb.msgFwd
              mb.msgFwd *= 0.0


    def gdd_update(self,maxstep=1.0,threshold=0.01):
        """gdd_update : update the clique factors using generalized dual decomposition gradients.
          Often slower than WMB messages, but should ensure monotonic bound changes.
        """
        from functools import reduce
        def wt_elim(f,w,pri):
            elim_ord = np.argsort( [pri[x] for x in f.vars] )
            tmp = f.copy();
            for i in elim_ord:
              tmp = tmp.lsePower([f.v[i]],1.0/w[i])
            return tmp
        def calc_bound( thetas, weights, pri):
            return sum([wt_elim(th,wt,pri) for th,wt in zip(thetas,weights)])
        def mu(th,w,pri):
            elim_ord = np.argsort( [pri[x] for x in th.vars] )
            lnZ0 = th
            lnmu = 0.0
            for i in elim_ord:
              lnZ1 = lnZ0.lsePower([th.v[i]],1.0/w[i])
              lnmu = lnmu + (lnZ0 - lnZ1)*(1.0/w[i])
              lnZ0 = lnZ1
            return lnmu.expIP()
        def armijo(thetas,weights,pri,steps,threshold=1e-4,direction=+1):
            f1 = calc_bound(thetas,weights,pri)
            match = reduce(lambda a,b: a&b, [th.vars for th in thetas], thetas[0].vars)
            for s in range(steps):
              mus = [mu(th,wt,pri).marginal(match) for th,wt in zip(thetas,weights)]
              dL = [mus[0]-mus[i] for i in range(len(mus))]
              dL[0] = -sum(dL)
              gradmag  = sum([ df.abs().sum() for df in dL ])
              gradmax  = max([ df.abs().max() for df in dL ])
              gradnorm = sum([ (df**2.0).sum() for df in dL ])
              if gradmax < 1e-8: return    # "optTol" : gradient small => local optimum  (use max(abs(g))?)
              stepsize = min(1.0, 1.0/gradmag) if s==0 else min(1.0, direction*(f0-f1)/gradmag)
              stepsize = stepsize if stepsize > 0 else 1.0
              f0 = f1;   # update "old" objective value
              for j in range(10):
                newthetas = [th+(direction*stepsize*df) for th,df in zip(thetas,dL)]   # redo; modify df directly
                f1 = calc_bound(newthetas,weights,pri)
                #print "  ",f0," => ",f1, "  (",f0-f1,' ~ ',stepsize*threshold*gradnorm,")"
                if (f0 - f1)*direction > stepsize*threshold*gradnorm:
                  for th,nth in zip(thetas,newthetas): th.t[:]=nth.t   # rewrite tables
                  break;
                else:
                  stepsize *= 0.5
                  if stepsize*gradmax < 1e-8: return  # < progTol => no progress possible

        #def armijo(thetas,weights,pri,maxstep,threshold,direction):
        #    f0 = calc_bound(thetas,weights,pri)
        #    #print [th for th in thetas], f0
        #    match = reduce(lambda a,b: a&b, [th.vars for th in thetas], thetas[0].vars)
        #    mus = [mu(th,wt,pri).marginal(match) for th,wt in zip(thetas,weights)]
        #    dL = [mus[0]-mus[i] for i in range(len(mus))]
        #    dL[0] = -sum(dL)
        #    gradnorm = sum([ (df**2.0).sum() for df in dL ])
        #    for j in range(10):
        #      newthetas = [th+(direction*maxstep*df) for th,df in zip(thetas,dL)]   # redo; modify df directly
        #      f1 = calc_bound(newthetas,weights,pri)
        #      #print "  ",f0," => ",f1, "  (",f0-f1,' ~ ',maxstep*threshold*gradnorm,")"
        #      if (f0 - f1)*direction > maxstep*threshold*gradnorm:
        #        for th,nth in zip(thetas,newthetas): th.t[:]=nth.t   # rewrite tables
        #        return
        #      else:
        #        maxstep *= 0.5
        #    return # give up?
        ######
        bound = 0.0
        for i,b in enumerate(self.buckets):
            for j,mb in enumerate(b):   # make sure has the correct scope (TODO: required?)
              if mb.theta.nvar < len(mb.clique): mb.theta += Factor(mb.clique - mb.theta.vars,0.0)
        for i,b in enumerate(self.buckets):
            X = self.model.vars[ self.elimOrder[i] ]
            nNodes = len(b)
            thetas = [mb.theta for mb in b]
            eps = 1e-3 * self.weights[i]    # TODO: doesn't work with mixed weight signs
            weights = [ [eps for x in mb.theta.vars] for mb in b ]
            for j,mb in enumerate(b): weights[j][mb.theta.vars.index(X)] = mb.weight
            armijo(thetas,weights,self.priority,5,threshold,np.sign(eps))
            for j,mb in enumerate(b):
              if mb.parent is not None:
                thetas2 = [mb.theta, mb.parent.theta]
                pi,pj = self.__nodeID(mb.parent)
                weights2 = [ weights[j], [1e-3*self.weights[pi] for x in mb.parent.theta.vars] ]
                weights2[1][mb.parent.theta.vars.index(self.model.vars[self.elimOrder[pi]])] = mb.parent.weight
                armijo(thetas2,weights2,self.priority,5,threshold,np.sign(eps))  # TODO: mixed?
              bound += calc_bound([mb.theta],[weights[j]],self.priority)
        return float(bound)




    # TODO: rename greedy-assign?  Add optional partial config?
    def assignBackward(self):
        """Perform a backward pass through all nodes, greedily assigning the most likely value.
        If weights='max' and each bucket has only one mini-bucket, this returns the optimal model configuration.
        """
        # TODO check & test  ; check for zero weights? (don't care?)
        x = {}
        for i,b in reverse_enumerate(self.buckets):    
            X = self.model.vars[ self.elimOrder[i] ]
            bel = Factor([X],0.0)
            for j,mb in enumerate(b):
                bel += mb.theta.condition(x)
                for c in mb.children: bel += c.msgFwd.condition(x)
            x[X] = bel.argmax()[0]
        return x

    def initHeuristic(self, pseudotree=None):
        """TODO: make this function unnecessary; make work for and/or pseudotree (currently or only)"""
        self.atElim = [ [] for b in self.buckets ]
        if pseudotree is not None: assert(np.all([x==x_ for x,x_ in zip(self.elimOrder,pseudotree.order)]))
        for i,b in enumerate(self.buckets):
          for j,mb in enumerate(b):
            if mb.parent is not None:
              pi,pj = self.__nodeID(mb.parent)
              ii_ = i+1 if pseudotree is None else self.priority[pseudotree.parent[self.elimOrder[i]]]
              for ii in range(i+1,pi+1): 
                  if pseudotree:           # if we have a pseudotree, follow its parent arrows down to pi
                      if ii<ii_: continue
                      else: 
                          if ii!=pi: ii_ = self.priority[pseudotree.parent[self.elimOrder[ii]]]
                  self.atElim[self.elimOrder[ii]].append(mb);

    def heuristic(self,X,config):
        """Evaluate the bound given partial assignment 'config' (including variable X and all later)"""
        return sum([mb.msgFwd.valueMap(config) for mb in self.atElim[X]])
        # need desired pseudo-tree & track messages passing between earlier & later buckets

    def resolved(self,X,config):
        """Evaluate the resolved value of a partial assignment 'config' (including variable X and all later)"""
        return sum([mb.theta.valueMap(config) for b in self.buckets[self.priority[X]:] for mb in b])

    def newly_resolved(self,X,config):
        """Evaluate the change in resolved value of a partial assignment 'config' after clamping X"""
        return sum([mb.theta.valueMap(config) for mb in self.buckets[self.priority[X]]])

    def sample(self):  
        """Draw a sample from the WMB-IS mixture proposal (assumes sum+ task)
        Returns x,q  where x (config) is the sampled configuration and q (float) is its log-probability under the proposal.
        """
        # TODO: add argument "x" for partial conditioning?  (return of configuration? or return tuple?)
        # TODO check for positive, unit sum weights?  (don't care?)
        x = {}
        logQx = 0.0
        for i,b in reverse_enumerate(self.buckets): 
            X = self.model.vars[ self.elimOrder[i] ]
            qi = Factor([X],0.0)
            for j,mb in enumerate(b):
                qij = mb.theta.condition(x)
                qij -= mb.msgFwd if mb.parent is None else mb.msgFwd.condition(x)
                for c in mb.children: qij += c.msgFwd.condition(x)
                qij -= qij.max()
                qij *= 1.0 / mb.weight
                qij.expIP()
                qij *= mb.weight
                qi += qij
            qi /= qi.sum()   # normalize (should be already)
            xval = qi.sample(Z=1.0)[0]
            x[X] = xval
            logQx += np.log( qi[xval] )
        return x,logQx


# functions:
# addClique(vs) : add clique etc and [fix up structure beneath]
#   => build = add each clique in turn => valid structure?
# 

# (0) basic merge op: join cliques & theta; fix parent; fix merges; fix msgFwd
# (1) "" + fix forward also
# (1) merge by score (score f'n: upper bd, scope, etc.)
#     
#  msgFwd(...)
#
#  msgBwd(...)
#
#  reparameterize()?  -> write into model?
#
#  heuristic?
#
#  bound / value
#
#  __str__ : wmbPrint
#
#  sample() : draw from mixture proposal (if positive weights?)
#
#

class JTree(WMB):
    """Junction tree object for exact inference"""
    def __init__(self, model, elimOrder=None, weights=1.0):
        super(JTree,self).__init__(model,elimOrder=elimOrder,weights=weights)
        self.merge(lambda a,b: 1.0)    # merge until exact 
        self.forwardDone = False
        self.setWeights(weights)

    def msgForward(self):
        """Perform a forward pass of the junction tree.
          Returns the inference value: partition function for `sum`, optimal value for `max`.
        """
        return_value = super(JTree,self).msgForward()
        self.forwardDone = True
        return return_value

    def beliefs(self, beliefs=None):
        """Compute a set of marginals of the junction tree.  Calls msgForward if not yet done.
          beliefs = list of variable sets, corresponding to the desired marginals.
          Returns a corresponding length list of marginal factors.
        """
        if beliefs is None: return {}
        if not self.forwardDone: self.msgForward()   # or raise valueerror?
        return super(JTree,self).msgBackward(beliefs=beliefs)

    def argmax(self):
        """Compute a maximizing argument of the junction tree.  Calls msgForward if not yet done."""
        if not self.forwardDone: self.msgForward()   # or raise valueerror?
        return super(JTree,self).assignBackward()

    def sample(self):
        """Draw a sample from the junction tree.  Calls msgForward if not yet done."""
        if not self.forwardDone: self.msgForward()   # or raise valueerror?
        return super(JTree,self).sample()

    def __repr__(self):
        """Overrides WMB __repr__ method to remove unnecessary details"""
        to_return = ""
        for i,b in enumerate(self.buckets):
            to_return += "{:03d}: ".format(int(self.elimOrder[i]))
            for j,mb in enumerate(b):
                to_return += "{!s} => ({}); ".format(mb, self._WMB__nodeID(mb.parent)[0])
            to_return += "\n"
        return to_return

    # TODO?: Smartly accommodate changes to the model?



