"""
pyGM/filetypes.py

Read / write methods for graphical model file types (UAI, WCSP, etc.)

readUai  /  writeUai            : read/write UAI competition file format
readEvidence10, readEvidence14  : read/write UAI, Ergo evidence format
readErgo                        : read Ergo Bayes net format
readWCSP                        : read WCSP weighted CSP format

Version 0.0.1 (2015-09-28)
(c) 2015 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import numpy as np;
from sortedcontainers import SortedSet as sset;
from pyGM.factor import *


def readFileByTokens(path, specials=[]):
  """Helper function for parsing pyGM file formats"""
  import re
  spliton = '([\s'+''.join(specials)+'])'
  with open(path, 'r') as fp:
    for line in fp:
      if line[-1]=='\n': line = line[:-1]
      tok = re.split(spliton,line)
      for t in tok: 
        t = t.strip()
        if t != '':
          yield t



def readUai(filename):
  """Read in a collection (list) of factors specified in UAI 2010 format

  Example:
  >>> factor_list = readUai10( 'path/filename.uai' )
  """
  dims = []           # store dimension (# of states) of the variables
  i = 0               # (local index over variables)
  cliques = []        # cliques (scopes) of the factors we read in
  factors = []        # the factors themselves
  evid = {}           # any evidence  (TODO: remove)

  gen = readFileByTokens(filename,'(),')   # get token generator for the UAI file
  type = next(gen)                   # read file type = Bayes,Markov,Sparse,etc
  nVar = int(next(gen))              # get the number of variables
  for i in range(nVar):              #   and their dimensions (states)
    dims.append( int(next(gen)) )
  nCliques = int(next(gen))          # get the number of cliques / factors
  for c in range(nCliques):          #   and their variables / scopes
    cSize = int(next(gen))           #   (size of clique)
    cliques.append([])
    for i in range(cSize):           #   ( + list of variable ids)
      v = int(next(gen))
      cliques[-1].append( Var(v,dims[v]) )
    #print cliques[-1]
  for c in range(nCliques):          # now read in the factor tables:
    tSize = int(next(gen))           #   (# of entries in table = # of states in scope)
    vs = VarSet(cliques[c])
    assert( tSize == vs.nrStates() )
    factors.append(Factor(vs))       # add a blank factor
    factorSize = tuple(v.states for v in cliques[c]) if len(cliques[c]) else (1,)
    tab = np.array([next(gen) for tup in range(tSize)],dtype=float,order='C').reshape(factorSize)
    t2  = np.transpose(tab, tuple(np.argsort([v.label for v in cliques[c]])))
    factors[-1].table = np.array(t2,dtype=float,order=orderMethod)   # use 'orderMethod' from Factor class

  used = np.zeros((nVar,))
  for f in factors: used[f.v.labels] = 1
  for i in range(nVar):              # fill in singleton factors for any missing variables
    if dims[i] > 1 and not used[i]: factors.append(Factor([Var(i,dims[i])],1.0))

  return factors

def readEvidence10(evidence_file):
  """Read UAI-2010 evidence file

  The 2010 specification allowed multiple evidence configurations in the same file.
  >>>  evList = readEvidence10('path/file.uai.evid') 

  Returns a list of evidence configurations; evList[i] is a dictionary, { Xi : xi , ... } 
  indicating that variable Xi = value xi.
  """
  # read evidence file: 2010 version (multiple evidences)
  gen = readFileByTokens(evidence_file)   # get token generator for the UAI file
  nEvid = int(next(gen))             # get number of evidences
  evid = []
  for i in range(nEvid):             # for each, create a blank dictionary
    evid.append( {} )
    ev_length = int(next(gen))       # read how many evidence variables
    for j in range(ev_length):       #   and then that many (id,value) pairs
      var,val = int(next(gen)), int(next(gen))
      evid[i][var]=val
  return evid  

  

def readEvidence14(evidence_file):
  """Read a UAI-2014 format evidence file

  The 2014 specification allowed only one evidence configuration per file.
  >>> ev = readEvidence14('path/file.uai.evid') 

  Returns an evidence configuration as a dictionary, { Xi : xi , ... }, 
  indicating that variable Xi = value xi.
  """
  # read evidence file: 2014 version (single evidence)
  gen = readFileByTokens(evidence_file)   # get token generator for the UAI file
  evid = {}                          # create blank dictionary
  ev_length = int(next(gen))         # read how many evidence variables
  for j in range(ev_length):         #   and then that many (id,value) pairs
    var,val = int(next(gen)), int(next(gen))
    evid[var]=val
  return evid  


def writeUai(filename, factors):
  """Write a list of factors to <filename> in the UAI competition format"""
  with open(filename,'w') as fp:
    fp.write("MARKOV\n")               # format type (TODO: add support)

    nvar = np.max( [np.max( factors[i].vars ) for i in range(len(factors))] ).label + 1
    fp.write("{:d}\n".format(nvar))    # number of variables in model

    dim = [0 for i in range(nvar)]     # get variable dimensions / # states from factors
    for f in factors:
      for v in f.vars:
        dim[v.label] = v.states
    fp.write(" ".join(map(str,dim)) + "\n")  # write dimensions of each variable
    fp.write("\n")                     # (extra line)
   
    fp.write("{:d}\n".format(len(factors))); # number of factors
    for f in factors:                  # + cliques
      fp.write("{:d} ".format(f.nvar) + " ".join(map(str,f.vars)) + "\n")
    fp.write("\n")                     # (extra line)
    for f in factors:                  # factor tables
      fp.write("{:d} ".format(f.numel()) + " ".join(map(str,f.t.ravel(order='C'))) + "\n")

 



# TODO: test
def readErgo(filename):
  """ Read in a Bayesian network (list of conditional probabilities) specified in ERGO format

  Example:
  >>> factor_list,names,labels = readErgo( 'path/filename.erg' )

  See e.g. http://graphmod.ics.uci.edu/group/Ergo_file_format for details
  """
  dims = []           # store dimension (# of states) of the variables
  i = 0               # (local index over variables)
  cliques = []        # cliques (scopes) of the factors we read in
  factors = []        # the factors themselves
  evid = {}           # any evidence  (TODO: remove)

  gen = readFileByTokens(filename)   # get token generator for the UAI file
  nVar = int(next(gen))              # get the number of variables
  for i in range(nVar):              #   and their dimensions (states)
    dims.append( int(next(gen)) )
  nCliques = nVar                    # Bayes net => one clique per variable
  for c in range(nCliques):          #   and their variables / scopes
    cSize = int(next(gen))           #   (number of parents)
    cliques.append([c])              #   (clique is Xc + parents)
    for i in range(cSize):           #   ( => read list of parent variable ids)
      v = int(next(gen))
      cliques[-1].append( Var(v,dims[v]) )
    #print cliques[-1]
  for c in range(nCliques):          # now read in the conditional probabilities
    tSize = int(next(gen))           #   (# of entries in table = # of states in scope)
    vs = VarSet(cliques[c])
    assert( tSize == vs.nrStates() )
    factors.append(Factor(vs))       # add a blank factor
    factorSize = tuple(v.states for v in cliques[c]) if len(cliques[c]) else (1,)
    #pi = list(map(lambda x:vs.index(x), cliques[c]))
    #ipi = list(pi)                   # get permutation mapping: file's order to sorted order
    #for j in range(len(pi)):         #   (ipi = inverse permutation)
    #  ipi[pi[j]] = j
    #print 'Building %s : %s,%s : %s'%(cliques[c],factorSize,vs,tSize)
    tab = np.array([next(gen) for tup in range(tSize)],dtype=float,order='C').reshape(factorSize)
    t2  = np.transpose(tab, tuple(np.argsort([v.label for v in cliques[c]])))
    factors[-1].table = np.array(t2,dtype=float,order=orderMethod)   # use 'orderMethod' from Factor class
    #
    #for tup in np.ndindex(factorSize):  # automatically uai order? ("big endian")
    #  tok = next(gen)
    #  #print "%s => %s: %s"%(tup,tuple(tup[ipi[j]] for j in range(len(ipi))),tok)
    #  if (tok == '('):               # check for "sparse" (run-length) representation
    #    run, comma, val, endparen = next(gen), next(gen), next(gen), next(gen)
    #    assert(comma == ',' and endparen==')')
    #    for r in range(run):         #   if so, fill run of table with value
    #      mytup = tuple(tup[ipi[j]] for j in range(len(ipi)))
    #      factors[-1][mytup] = float(val)
    #  else:                          # otherwise just a list of values in the table
    #    mytup = tuple(tup[ipi[j]] for j in range(len(ipi)))
    #    factors[-1][mytup] = float(tok)

  names,labels = [],[]

  for i in range(nVar):
    names.append( str(next(gen)) )

  for i in range(nVar):
    labels.append( [] )
    for j in range(dims[i]):
      labels[-1].append( str(next(gen)) )
  
  return factors,names,labels





# TODO: test
def readWCSP(filename):
  """ Read in a weighted CSP (list of neg-log factors) specified in WCSP format
  
  Example:
  >>> factor_list,name,upperbound = readWCSP( 'path/filename.wcsp' )
 
  See e.g. http://graphmod.ics.uci.edu/group/WCSP_file_format
  """
  gen = readFileByTokens(filename)   # get token generator for the UAI file
  name = str(next(gen))
  nVar = int(next(gen))
  max_domain = int(next(gen))
  nConstraints = int(next(gen))
  ub = int(next(gen)) 

  dims = []
  for i in range(nVar):
    dims.append( int(next(gen)) )
  
  constraints = []
  for c in range(nConstraints):
    cSize = int(next(gen))             # size of the clique, then clique variables
    cVars = []
    for i in range(cSize):
      xi = int(next(gen))
      cVars.append( Var(xi,dims[xi]) )
    defaultCost = float(next(gen))     # default cost for non-specified tuples

    vs = VarSet(cVars)
    constraints.append( Factor(vs, defaultCost) )
    pi = list(map(lambda x:vs.index(x), cVars))
    ipi = list(pi)                   # get permutation mapping: file's order to sorted order
    for j in range(len(pi)):         #   (ipi = inverse permutation)
      ipi[pi[j]] = j

    nTuples = int(next(gen))           # of non-default-value tuples
    for j in range(nTuples):
      tup = tuple( int(next(gen)) for i in range(cSize) )   # read tuple
      mytup = tuple( tup[ipi[j]] for j in range(len(ipi)) ) # convert to sorted order
      constraints[-1][mytup] = float(next(gen))                 # and update factor
      
  return constraints,name,ub



# TODO: test
def writeWCSP(filename, factors):
  """Write 'filename' in weighted CSP format 

  (see http://graphmod.ics.uci.edu/group/WCSP_file_format)
  TODO: exploit sparsity (use most common value in table)
  """
  with open(filename,'w') as fp:
    nvar = np.max( [np.max( factors[i].vars ) for i in range(len(factors))] ).label + 1
    dmax = np.max( [np.max( factors[i].dims() ) for i in range(len(factors))] )
    ub   = np.sum( [factors[i].max() for i in range(len(factors)) ] )
    default_value = 0
    
    dim = [0 for i in range(nvar)]     # get variable dimensions / # states from factors
    for f in factors:
      for v in f.vars:
        dim[v.label] = v.states

    fp.write(filename)                 # write preamble: name, #var, max_dim, #constraints, upper-bound
    fp.write(" {:d} {:d} {:d} {:d}\n".format(nvar,dmax,len(factors),ub))
    fp.write(" ".join(map(str,dim)) + "\n")  # write dimensions of each variable
    for f in factors:                  # write each factor:
      fp.write("{:d} ".format(len(f.vars)))
      for v in f.vars:                 # first the variable IDs in the factor
        fp.write("{:d} ".format(v.label))
      fp.write("{:d}\n".format(default_value))  # then the default vaule (unused)
      factorSize = f.dims() if f.nvar() else (1,)
      for tup in np.ndindex(factorSize):  # then the value of each tuple
        fp.write(" ".join(map(str,tup)))
        fp.write(" {:d}\n".format(f[tup]))


def readLimid(filename):
  """Read in a LIMID file (Maua format)

  Example: get CPTs for chance nodes C, uniform policies D, and utilities U:
  >>>  C,D,U = readLimid(filename) 

  See e.g. https://github.com/denismaua/kpu-pp
  TODO: may have an error in variable orderings?  Hard to tell from Denis' page.
  TODO: seems to expect multiplicative utility functions?
  """
  dims = []           # store dimension (# of states) of the variables
  i = 0               # (local index over variables)
  cliques = []        # cliques (scopes) of the factors we read in
  factors = []        # the factors themselves

  gen = readFileByTokens(filename)   # get token generator for the UAI file
  type = str(next(gen))
  if type=='/*': 
    while type[-2:]!='*/': type = str(next(gen))
    type = str(next(gen))
  if type != 'LIMID': raise ValueError('Not LIMID file?')
  nC = int(next(gen))
  nD = int(next(gen))
  nU = int(next(gen))

  for i in range(nC+nD):             #   and their dimensions (states)
    dims.append( int(next(gen)) )
  nCliques = nC + nD + nU            # one CPT per chance node 
  for c in range(nCliques):          #
    cSize = int(next(gen))           # number of parents
    if c < nC + nD: 
      cliques.append([Var(c,dims[c])])  # CPT for chance node c or decision node d; else utility
    else:
      cliques.append([])             
    for i in range(cSize):           #   get list of parents
      v = int(next(gen))
      cliques[-1].append( Var(v,dims[v]) )
    cliques[-1] = list(reversed(cliques[-1]))   # !!!! can't tell if this is right !!!
    #print cliques[-1]
  factors = [None]*(nCliques)
  for c in range(nC,nC+nD): factors[c] = Factor(cliques[c],1.0/dims[c]);  # uniform initial policy
  CpU = range(nC)
  CpU.extend(range(nC+nD,nCliques))
  for c in CpU:                      # now read in the factor tables:
    tSize = int(next(gen))           #   (# of entries in table = # of states in scope)
    vs = VarSet(cliques[c])
    #print cliques[c], ' => ', vs
    assert( tSize == vs.nrStates() )
    factors[c] = Factor(vs)        # add a blank factor
    factorSize = tuple(v.states for v in cliques[c]) if len(cliques[c]) else (1,)
    tab = np.array([next(gen) for tup in range(tSize)],dtype=float,order='C').reshape(factorSize)
    t2  = np.transpose(tab, tuple(np.argsort([v.label for v in cliques[c]])))
    factors[c].table = np.array(t2,dtype=float,order='F')

  return factors[:nC],factors[nC:nC+nD],factors[nC+nD:]



def Limid2MMAP(C,D,U):
  """Convert LIMID factors into MMAP factors & query variable list  (Not Implemented)

  Example:
  >>> factors,query = Limid2MMAP(C,D,U) 

  See also readLimid().

  TODO: add additive utility transformation?  Maua spec seems to be multiplicative?
  """
  nC,nD,nU = len(C),len(D),len(U)
  nV = nC+nD
  X = [None]*nV
  for f in C+D: 
    for v in f.vars:
      X[v.label] = v
  nxt = nV
  DD = []
  Q  = []
  for d in range(nD):
    dd = nC+d
    par = D[d].vars - [dd]
    #print "Processing ",dd," parents ",par
    if par.nrStates()==1:   # no parents => decision variable = map variable
      Q.append(dd)
      continue              # otherwise, create one map var per config, for that policy
    for p in range(par.nrStates()):
      X.append( Var(nxt,X[dd].states) )   # create MAP var for policy
      #print "  new query var ",nxt
      cliq = [v for v in par]
      cliq.extend( [X[dd],X[nxt]] )
      tab = np.ones( (par.nrStates(),) + (X[dd].states,X[dd].states) )
      tab[p,:,:] = np.eye(X[dd].states)
      tab = np.squeeze(tab)
      DD.append( Factor(VarSet(cliq), np.transpose(tab, tuple(np.argsort([v.label for v in cliq]))) ) )
      Q.append(nxt)
      nxt += 1
  return C+DD+U, Q
    



def readOrder(filename):
    """Read an elimination order from a file

    Elimination orders are stored as unknown length vectors, format "[nvar] [v0] [v1] ... [vn]"

    Note: the same file format may also be useful for MPE configurations, etc.
    """
    with open(filename,'r') as fp:
        lines = fp.readlines();
    text = lines[-1].strip('\n').split(' ');
    nvar = int(text[0]);
    vals = [int(text[i]) for i in range(1,nvar+1)];
    if len(vals) != nvar: raise ValueError("Problem with file?");
    return vals



def writeOrder(filename,order):
    """ Write an elimination order (or other vector) to a file """
    with open(filename,'w') as fp:
        fp.write("{} ".format(len(order)));
        fp.write(" ".join(map(str,order)));
  



