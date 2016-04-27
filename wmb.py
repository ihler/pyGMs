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



class WMB(object):
    '''Class implementing weighted mini-bucket elimination inference'''

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


    class ConstantList:
        def __init__(self, val):
            self.val = val
        def __getitem__(self, loc):
            return self.val


    def __init__(self, model, elimOrder=None, iBound=0, sBound=0, weights=1.0, **kwargs):
        # save a reference to our model
        self.model = model

        # create & process elimination ordering of the model:
        if elimOrder is None: elimOrder = 'wtminfill'
        if type(elimOrder) is str:
            # TODO: check that weights is string or float?
            raise NotImplementedError  # TODO: call elimination order with method
        self.elimOrder = elimOrder
        self.priority = [-1 for i in range(model.nvar)]  # build priority of each var
        for i,x in enumerate(elimOrder): self.priority[x] = i

        # now build the mini-bucket data structure
        self.buckets = [ [] for x in range(model.nvar) ]  # bucket for each var: list of minibuckets 
        self.matches = [ [] for x in range(model.nvar) ] # matching sets for each bucket
        for f in model.factors:
            n = self.addClique(f.vars)
            n.theta += f.log()                 # include log f(x) in node's log-factor
            n.originals.append(f)              # append (pointer to) original f for later reference
        # and set the weights of the buckets:
        self.setWeights(weights)
 
    def setWeights(self,weights):
        """Set the weights of the inference problem.
        weights = 'max+' or 0.0  => upper bound the MAP configuration
                  'sum+' or 1.0  => upper bound the partition function
                  'sum-' or -1.0 => lower bound the partition function
        For more general bounds, weights = list of floats (one per variable)
        """
        if type(weights) is str:
            if weights == 'sum+': weights = 1.0
            elif weights=='sum-': weights = -1.0
            elif weights=='max+': weights = 0.0
            else: raise ValueError("Unknown weight / task type; must be max+, sum+, sum-, or float / float list")
        if type(weights) is float: weights = WMB.ConstantList(weights)
        self.weights = weights

        for i,xi in enumerate(self.elimOrder):     # (TODO?) set mini-bucket weights uniformly
            ni = len(self.buckets[i])
            for j in range(ni):               # set uniformly
                self.buckets[i][j].weight = self.weights[xi]/ni
            if self.weights[xi] < 0:          # uniform for lower bound:
                self.buckets[i][0].weight = 1.0 - self.weights[xi]*(ni-1)/ni



 
    def __nodeID(self,node):
        """Helper function: get identifier (bucket & location) of a given node""" 
        if not isinstance(node, WMB.Node): return None,None
        i = min([self.priority[x] for x in node.clique])  # get bucket
        j = self.buckets[i].index(node)
        return i,j
   
    def __repr__(self):     
        to_return = ""
        for i,b in enumerate(self.buckets):
            to_return += "{:03d}: ".format(int(self.elimOrder[i]))
            for j,mb in enumerate(b):
                to_return += "{!s} => {}; ".format(mb, self.__nodeID(mb.parent))
            to_return += "\n"
        return to_return
        

    def addClique(self,vars):
        """Add a clique with scope "vars", fixing up structure to be a valid MB tree"""
        vs = VarSet(vars)
        corder = np.argsort( [self.priority[x] for x in vars]  ) # get order in which eliminated
        corder = [vars[c] for c in corder]
        added = []
        found = False
        for x in corder:
            if found: break
            #print "bucket ",x
            b = self.buckets[self.priority[x]]
            to_remove = []
            for mb in b:
                #print "  check ",mb
                if mb.clique < vs:
                   to_remove.append(mb)
                if mb.clique >= vs:                # if we found a minibucket we can just join, do:
                    if len(added) > 0:             # if we've added nodes, connect them as descendants
                        mb.children.append( added[-1] )  # of the found node, and found node as parent of last
                        added[-1].parent = mb          
                    found = True                   # now, we don't need to keep generating parents
                    #print "    Found!"
                    added.append(mb)               # not really added, but the end of the added chain
                    break
            # if we didn't find any mini-buckets we can join, we need to add one:
            if not found:                          # 
                n = WMB.Node()
                n.clique = VarSet(vs)
                n.weight = 0.0
                #print "adding ",n," to ",self.priority[x]
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
                # TODO: fix up match structure?
        # done adding all required cliques; return 1st 
        return added[0]


    # score = len(max)+len(min)/len(max) if union < iBound else -1 for scope
    def merge(self, score):
        from heapq import heappush,heappop
        for b in self.buckets:
            priority = [] 
            lookup = {}
            # see not-too-efficient-looking remove_task https://docs.python.org/2/library/heapq.html
            # build scores:
            for i,m1 in enumerate(b):
                for m2 in b[i+1:]:
                    s = score(m1,m2)
                    if s >= 0.0:
                        entry = [-s,m1,m2]
                        lookup[(m1,m2)] = entry
                        heappush(priority, entry)
            while len(priority):
                entry = heappop(priority)
                s,m1,m2 = entry[0],entry[1],entry[2]
                #s,m1,m2 = priority.pop()
                if m1 is None or m2 is None: continue
                if m1 not in b or m2 not in b: continue   ## check for removed minibuckets?
                #print b
                #print "Merging ",m1,"+",m2
                for m in b:
                    for ma,mb in [(m1,m),(m,m1),(m2,m),(m,m2)]:
                        s = -lookup.get( (ma,mb), [1,None,None] )[0]
                        if s >= 0.0: 
                            entry = lookup.pop( (ma,mb) )
                            entry[1],entry[2] = None,None  # mark as "removed"
                            #priority.remove( [s,ma,mb] )
                m12 = self.addClique( m1.clique | m2.clique )
                # what if others removed?  (bad check above?)
                #print b
                for m in b:
                    if m is m12: continue
                    s = score(m12,m)
                    if s >= 0.0:
                        entry = [-s,m12,m] 
                        lookup[ (m12,m) ] = entry
                        heappush(priority,entry)
        return None



    def mergeScope(iBound=0, sBound=0): 
        for Bi in self.buckets:
            # TODO: sort bucket by size (or store sorted)
            for mb in Bi:
                # TODO: merge into largest clique that can fit
                pass

    #@profile
    def msgForward(self, stepTheta=0.5, stepWeights=0.1):
        """Compute a forward pass through all nodes and return the resulting bound"""
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
                        for j,mb in enumerate(match):
                            delta[j] = beliefs[j].lse( mb.clique - vAll )  # TODO: belief[j] incorrect if match != b
                            avg_belief += delta[j] * mb.weight / wTotal
                        for j,mb in enumerate(match):
                            delta[j] = avg_belief - delta[j]
                            beliefs[j] += delta[j] * stepTheta
                            beliefs[j] -= beliefs[j].lse()
                            mb.theta += delta[j] * mb.weight * stepTheta
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
                mb.msgFwd = beliefs[j].lsePower([X], 1.0/mb.weight)
                beliefs[j] = Factor().log() # clear reference & memory? 
                if mb.parent is None:       # add roots to overall bound
                    bound += mb.msgFwd  
        return float(bound)


    #@profile
    def msgBackward(self, stepTheta=0.0, stepWeights=0.0, beliefs=None):
        """Compute a backward pass through all nodes
           If beliefs is a list of cliques, returns the estimated beliefs on those cliques
        """
        to_save = [ [] ]*len(self.buckets)
        if beliefs is None:
            return_beliefs = {}
        else:
            return_beliefs = { tuple(clique): None for clique in beliefs }
            # map cliques to buckets for checking
            for clique in beliefs:
                to_save[ min([self.priority[x] for x in clique]) ].append(VarSet(clique))
        for i,b in reversed(list(enumerate(self.buckets))):
            X = self.model.vars[ self.elimOrder[i] ]
            nNodes = len(b)
            beliefs_b = [ None for mb in b ]
            if nNodes > 1:  # if more than one mini-bucket partition, pre-compute beliefs:
                if stepTheta or stepWeights:
                    for j,mb in enumerate(b):
                        beliefs_b[j] = mb.theta + mb.msgBwd
                        for c in mb.children: beliefs_b[j] += c.msgFwd
                        beliefs_b[j] *= 1.0/mb.weight
                        beliefs_b[j] -= bel[j].lse()
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
                beliefs_b[j] -= beliefs_b[j].max()
                beliefs_b[j] *= 1.0 / mb.weight
                for c in mb.children:
                    c.msgBwd = beliefs_b[j].lse( mb.clique - c.clique )*c.weight - c.msgFwd
                    #c.msgBwd = (beliefs_b[j]*(1.0/mb.weight)).lse( mb.clique - c.clique )*c.weight - c.msgFwd
                # TODO: compute marginal of any to_save[i] cliques that fit & not done
                for c in to_save[i]:
                    if c <= mb.clique and return_beliefs[tuple(c)] is None: return_beliefs[tuple(c)] = beliefs_b[j].lse( mb.clique - c )
                beliefs_b[j] = Factor().log() # clear out belief
        for c,f in return_beliefs.items(): 
            f -= f.lse()
            f.expIP()   # exponentiate and normalize beliefs before returning
            #f /= f.sum()
        return return_beliefs


    # TODO: rename greedy-assign?  Add optional partial config?
    def assignBackward(self):
        """Perform a backward pass through all nodes, assigning the most likely value"""
        # TODO check & test  ; check for zero weights? (don't care?)
        x = {}
        for i,b in reversed(list(enumerate(self.buckets))):
            X = self.model.vars[ self.elimOrder[i] ]
            bel = Factor(X,0.0)
            for j,mb in enumerate(b):
                bel += mb.theta.condition(x)
                for c in mb.children: bel += c.msgFwd.condition(x)
            x[X] = bel.argmax()[0]
        return x

    def heuristic(self,x):
        """Evaluate the bound given partial assignment x"""
        raise NotImplementedError   # TODO: fix
        # need desired pseudo-tree & track messages passing between earlier & later buckets

    def sample(self):  
        """Draw a sample from the WMB-IS mixture proposal (assumes sum+ task)"""
        # TODO: add argument "x" for partial conditioning?  (return of configuration? or return tuple?)
        # TODO check for positive, unit sum weights?  (don't care?)
        x = {}
        logQx = 0.0
        for i,b in reversed(list(enumerate(self.buckets))):
            X = self.model.vars[ self.elimOrder[i] ]
            qi = Factor(X,0.0)
            for j,mb in enumerate(b):
                qij = mb.theta.condition(x)
                qij -= mb.msgFwd if mb.parent is None else mb.msgFwd.condition(x)
                for c in mb.children: qij += c.msgFwd.condition(x)
                qij -= qij.max()
                qij *= 1.0 / mb.weight
                qij.expIP()
                #if qij.sum() == 0.0: qij += 1.0/qij.numel()  # TODO: safety check needed?
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

    def msgForward(self):
        return_value = super(JTree,self).msgForward()
        self.forwardDone = True
        return return_value

    def beliefs(self, beliefs=None):
        if beliefs is None: return {}
        if not self.forwardDone: self.msgForward()   # or raise valueerror?
        return super(JTree,self).msgBackward(beliefs=beliefs)

    def argmax(self):
        if not self.forwardDone: self.msgForward()   # or raise valueerror?
        return super(JTree,self).assignBackward()

    def sample(self):
        if not self.forwardDone: self.msgForward()   # or raise valueerror?
        return super(JTree,self).sample()

    # Smartly accommodate changes to the model?



