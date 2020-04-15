"""
pyGM/filetypes.py

Read / write methods for graphical model file types (UAI, WCSP, etc.)

readUai  /  writeUai            : read/write UAI competition file format
readEvidence10, readEvidence14  : read/write UAI, Ergo evidence formats (14: single evidence)
readErgo                        : read Ergo Bayes Net format
readWCSP /  writeWCSP           : read/write WCSP weighted CSP format

Version 0.0.1 (2015-09-28)
(c) 2015 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import numpy as np;
from sortedcontainers import SortedSet as sset;
from pyGM.factor import *
from builtins import range



def readFileByTokens(path, specials=[]):
  """Helper function for parsing pyGM file formats"""
  import re
  spliton = '([\s'+''.join(specials)+'])'
  with open(path, 'r') as fp:
    for line in fp:
      #if line[-1]=='\n': line = line[:-1]
      tok = [t.strip() for t in re.split(spliton,line) if t and not t.isspace()]
      for t in tok: yield t
        #t = t.strip()
        #if t != '':
        #  yield t


def stripComments(gen, start=['/*'],end=['*/']):
  while True:
    t = next(gen)
    if t not in start: 
      yield t
    else:
      while t not in end: 
        t = next(gen) 


################################################################################################
# Temporary functions for testing file parse speed, etc.
################################################################################################
def readFileByTokensNEW(path, specials=[]):
  """Helper function for parsing pyGM file formats"""
  with open(path,'r') as fp:
    tok0 = fp.readline()
    yield tok0
    txt = fp.read()
  #for t in txt.split(): yield t
  for t in map(float, txt.split()): yield t




def readTEST(filename):
  """Read in a collection (list) of factors specified in UAI (2006-?) format

  Example:

  >>> factor_list = readUai( 'path/filename.uai' )
  """
  dims = []           # store dimension (# of states) of the variables
  i = 0               # (local index over variables)
  cliques = []        # cliques (scopes) of the factors we read in
  factors = []        # the factors themselves

  gen = readFileByTokens(filename,'(),')   # get token generator for the UAI file
  type = next(gen)                   # read file type = Bayes,Markov,Sparse,etc
  nVar = int(next(gen))              # get the number of variables
  dims = [int(next(gen)) for i in range(nVar)] #   and their dimensions (states)
  nCliques = int(next(gen))          # get the number of cliques / factors
  cliques = [ None ] * nCliques
  for c in range(nCliques): 
    cSize = int(next(gen))           #   (size of clique)
    cliques[c] = [int(next(gen)) for i in range(cSize)]
  factors = [ None ] * nCliques 
  for c in range(nCliques):          # now read in the factor tables:
    tSize = int(next(gen))           #   (# of entries in table = # of states in scope)
    vs = VarSet([Var(v,dims[v]) for v in cliques[c]])
    assert( tSize == vs.nrStates() )
    factorSize = tuple(dims[v] for v in cliques[c]) if len(cliques[c]) else (1,)
    tab = np.array([next(gen) for tup in range(tSize)],dtype=float,order='C').reshape(factorSize)
    tab = np.transpose(tab, tuple(np.argsort(cliques[c])))
    factors[c] = Factor(vs, np.array(tab,dtype=float,order=orderMethod))   # use 'orderMethod' from Factor class

  used = np.zeros((nVar,))
  for f in factors: used[f.v.labels] = 1
  for i in range(nVar):              # fill in singleton factors for any missing variables
    if dims[i] > 1 and not used[i]: factors.append(Factor([Var(i,dims[i])],1.0))

  return factors


def readTEST2(filename):
  with open(filename,'r') as fp:
    fp.readline()
    txt = fp.read()
  data = map(float, txt.split())
  return data


def readTEST3(filename):
  with open(filename,'rb') as fp:
    filetype = fp.readline()
    data = np.fromfile(fp,sep=' ')
  nVar = int(data[0])
  dims = data[1:nVar+1].astype('int') 
  nCliques,gen = int(data[nVar+1]), nVar+2
  cliques = [ None ] * nCliques
  for c in range(nCliques):
    cSize = int(data[gen])           #   (size of clique)
    cliques[c] = data[gen+1:gen+cSize+1].astype('int') 
    gen = gen+cSize+1
  factors = [ None ] * nCliques
  for c in range(nCliques):          # now read in the factor tables:
    tSize = int(data[gen])           #   (# of entries in table = # of states in scope)
    vs = VarSet([Var(v,dims[v]) for v in cliques[c]])
    assert( tSize == vs.nrStates() )
    factorSize = tuple(dims[cliques[c]]) if len(cliques[c]) else (1,)
    tab = data[gen+1:gen+tSize+1].reshape(factorSize)
    gen = gen + tSize+1
    #tab = np.array([next(gen) for tup in range(tSize)],dtype=float,order='C').reshape(factorSize)
    tab = np.transpose(tab, tuple(np.argsort(cliques[c])))
    #factors[c] = Factor(vs, np.array(tab,dtype=float,order=orderMethod))   # use 'orderMethod' from Factor class
    factors[c] = Factor(vs, tab)   # use 'orderMethod' from Factor class
  
  used = np.zeros((nVar,))
  for f in factors: used[f.v.labels] = 1
  for i in range(nVar):              # fill in singleton factors for any missing variables
    if dims[i] > 1 and not used[i]: factors.append(Factor([Var(i,dims[i])],1.0))

  return factors




  

################################################################################################
# 
################################################################################################



def readUai(filename):
  """Read in a collection (list) of factors specified in UAI (2006-?) format

  Example:

  >>> factor_list = readUai( 'path/filename.uai' )
  """
  dims = []           # store dimension (# of states) of the variables
  i = 0               # (local index over variables)
  cliques = []        # cliques (scopes) of the factors we read in
  factors = []        # the factors themselves

  gen = readFileByTokens(filename,'(),')   # get token generator for the UAI file
  type = next(gen)                   # read file type = Bayes,Markov,Sparse,etc
  nVar = int(next(gen))              # get the number of variables
  dims = [int(next(gen)) for i in range(nVar)] #   and their dimensions (states)
  nCliques = int(next(gen))          # get the number of cliques / factors
  cliques = [ None ] * nCliques
  for c in range(nCliques): 
    cSize = int(next(gen))           #   (size of clique)
    cliques[c] = [int(next(gen)) for i in range(cSize)]
  factors = [ None ] * nCliques 
  for c in range(nCliques):          # now read in the factor tables:
    tSize = int(next(gen))           #   (# of entries in table = # of states in scope)
    vs = VarSet(Var(v,dims[v]) for v in cliques[c])
    assert( tSize == vs.nrStates() )
    factorSize = tuple(dims[v] for v in cliques[c]) if len(cliques[c]) else (1,)
    tab = np.empty(tSize)
    for i in range(tSize): tab[i]=float(next(gen))
    tab = tab.reshape(factorSize)
    t2  = np.transpose(tab, tuple(np.argsort(cliques[c])))
    factors[c] = Factor(vs, np.array(t2,dtype=float,order=orderMethod))   # use 'orderMethod' from Factor class

  used = np.zeros((nVar,))
  for f in factors: used[f.v.labels] = 1
  for i in range(nVar):              # fill in singleton factors for any missing variables
    if dims[i] > 1 and not used[i]: factors.append(Factor([Var(i,dims[i])],1.0))

  return factors



def readEvidence10(evidence_file):
  """Read UAI-2010 evidence file

  The 2010 specification allowed multiple evidence configurations in the same file:

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

  The 2014 specification allowed only one evidence configuration per file:

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
  #
  gen = stripComments(readFileByTokens(filename), ['/*'],['*/'])   # get token generator w/ comments
  nVar = int(next(gen))              # get the number of variables
  dims = [int(next(gen)) for i in range(nVar)] #   and their dimensions (states)
  nCliques = nVar                    # Bayes net => one clique per variable
  cliques = [ None ] * nCliques
  for c in range(nCliques):         #   and their variables / scopes
    cSize = int(next(gen))           #   (number of parents)
    cliques[c] = [int(next(gen)) for i in range(cSize)]+[c] # (clique is Xc + parents)
  factors = [ None ] * nCliques
  for c in range(nCliques):         # now read in the conditional probabilities
    tSize = int(next(gen))           #   (# of entries in table = # of states in scope)
    vs = VarSet(Var(v,dims[v]) for v in cliques[c])
    assert( tSize == vs.nrStates() )
    tab = np.empty(tSize)
    for i in range(tSize): tab[i]=float(next(gen))
    factorSize = tuple(dims[v] for v in cliques[c]) if len(cliques[c]) else (1,)
    tab = tab.reshape(factorSize)
    t2  = np.transpose(tab, tuple(np.argsort(cliques[c])))
    factors[c] = Factor(vs, np.array(t2,dtype=float,order=orderMethod))   # use 'orderMethod' from Factor class
  #
  names,labels = [],[]
  for i in range(nVar):
    names.append( str(next(gen)) )
  for i in range(nVar):
    labels.append( [] )
    for j in range(dims[i]):
      labels[-1].append( str(next(gen)) )
  #
  return factors,names,labels




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



def writeWCSP(filename, factors, upper_bound=None, use_float=False):
  """Write 'filename' in weighted CSP format 

  (see http://graphmod.ics.uci.edu/group/WCSP_file_format)
  """
  #DONE: exploit sparsity (use most common value in table)
  from collections import Counter
  with open(filename,'w') as fp:
    nvar = int(np.max( [np.max( factors[i].vars ) for i in range(len(factors))] ).label + 1)
    dmax = int(np.max( [np.max( factors[i].dims() ) for i in range(len(factors))] ))
    if upper_bound is None:
      ub   = int(np.ceil(np.sum( [factors[i].max() for i in range(len(factors)) ] )))
    else:
      ub   = upper_bound
    
    dim = [0 for i in range(nvar)]     # get variable dimensions / # states from factors
    for f in factors:
      for v in f.vars:
        dim[v.label] = v.states

    fp.write(filename)                 # write preamble: name, #var, max_dim, #constraints, upper-bound
    fp.write(" {:d} {:d} {:d} {:d}\n".format(nvar,dmax,len(factors),ub))
    fp.write(" ".join(map(str,dim)) + "\n")  # write dimensions of each variable
    for f in factors:                  # write each factor:
      fp.write("{:d} ".format(len(f.vars)))
      cnt = Counter(f.table.ravel())
      default_value,n_default = cnt.most_common(1)[0]
      for v in f.vars:                 # first the variable IDs in the factor
        fp.write("{:d} ".format(v.label))
      if use_float: fp.write("{:f} ".format(default_value))      # then the default vaule
      else:         fp.write("{:d} ".format(int(default_value))) #  (float or int)
      fp.write("{:d}\n".format(f.vars.nrStates()-n_default))  # number of non-default values
      for tup in np.ndindex(f.dims()):          # then the value of each tuple
        if f[tup] != default_value:
          fp.write(" ".join(map(str,tup)))
          if use_float: fp.write(" {:f}\n".format(f[tup]))
          else:         fp.write(" {:d}\n".format(int(f[tup])))


def readDSL(filename):
  """Read GeNIe XML DSL format Bayesian network"""
  # TODO: check table entry order
  import xml.etree.ElementTree as ET
  tree = ET.parse(filename)
  root = tree.getroot()
  nodes = root.findall('nodes')[0]   # extract list of nodes in the model
  X = {}
  F,D,U,names,labels = [],[],[],[],[]
  for node in list(nodes):     # get all the variable def's
    if node.tag == 'cpt' or node.tag == 'decision':  # cpts & decisions define a variable:
      name = node.attrib['id']
      states = node.findall('state')
      print("{} ({}): {} states".format(name,len(X),len(states)))
      X[name] = Var(len(X), len(states)) 
      names.append(name)
      labels.append([s.attrib['id'] for s in states])
    # get parents:
    par  = node.findall('parents')
    clique  = [] if len(par)==0 else [X[pname] for pname in par[0].text.split()]
    if node.tag == 'cpt' or node.tag == 'decision': clique = clique + [X[name]]
    # populate lists:
    if node.tag == 'cpt':
      factorSize = tuple(v.states for v in clique)
      vals  = node.findall('probabilities')[0]
      tab = np.array([float(f) for f in vals.text.split()]).reshape(factorSize)
      tab = np.transpose(tab, tuple(np.argsort([v.label for v in clique])))
      F.append( Factor(VarSet(clique),tab) )
    elif node.tag == 'decision':
      D.append( clique )
    elif node.tag == 'utility':
      factorSize = tuple(v.states for v in clique)
      vals  = node.findall('utilities')[0]
      tab = np.array([float(f) for f in vals.text.split()]).reshape(factorSize)
      tab = np.transpose(tab, tuple(np.argsort([v.label for v in clique])))
      U.append( Factor(VarSet(clique),tab) )
  # return model
  return F,D,U,names,labels



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
  """Convert LIMID factors into MMAP factors & query variable list

  Example:

  >>> C,D,U = readLimid(filename)
  >>> factors,query = Limid2MMAP(C,D,U) 

  See also readLimid().
  """
  nC,nD,nU = len(C),len(D),len(U)
  from .graphmodel import GraphModel
  model = GraphModel( C + [Factor(d,1.) for d in D] )  # "base" model of chance & decision f'ns
  X = [None] * (max([x.label for x in model.X])+1)
  for x in model.X: X[x.label]=x
  nxt = len(X)
  DD,Q = [],[]

  for d in D:
    par, dd = VarSet(d[:-1]), d[-1]   # decision var is last in list
    if par.nrStates()==1:   # no parents => decision variable = map variable
      #DD.append( D[d] )     # don't need to keep factor; do mark variable in query
      Q.append(dd.label)
      continue              # otherwise, create one map var per config, for that policy
    for p in range(par.nrStates()):
      X.append( Var(nxt,X[dd].states) )           # create MAP var for policy
      cliq = [v for v in par] + [X[dd],X[nxt]]    # factor includes parents, policy var, and final decision
      tab = np.ones( (par.nrStates(),) + (X[dd].states,X[dd].states) )
      tab[p,:,:] = np.eye(X[dd].states)           # for par config p, final decision = policy var; ow anything OK
      tab = tab.reshape( par.dims() + (X[dd].states,X[dd].states) )
      tab = np.squeeze(tab)                       # now build the factor (may require permuting axes)
      DD.append( Factor(VarSet(cliq), np.transpose(tab, tuple(np.argsort([v.label for v in cliq]))) ) )
      Q.append(nxt)                               # policy var is part of the MMAP query
      nxt += 1
  if True: #latent == 'joint':
    z = Var(nxt, nU)
    UU = [None]*nU
    #fZ = Factor(z, 1.)
    for i,u in enumerate(U):
      if (u.min() < 0): raise ValueError("Utility {} has negative values!".format(str(u)))
      UU[i] = Factor( [v for v in u.vars]+[z], 1. )   # go thru constructor to ensure orderMethod match
      newarr = UU[i].table.reshape( (u.vars.nrStates(),nU) )
      newarr[:,i] = u.table.ravel()
      UU[i] = Factor( [v for v in u.vars]+[z], newarr.reshape(UU[i].table.shape) )
  else:   # latent == 'chain' ??
    for u in range(nU):
      # TODO: complete "chain" version of latent variable
      zi = Var(nxt,2)
      # add factor p(zi|zi-1)*U
      
  return C+DD+UU, Q
    

def readLimidCRA(filename):
  """Read in a LIMID file specified by our format with Charles River.

  Example:

  >>> fChance,fDecision,fUtil = readLimidCRA( 'path/filename.uai' )
  """
  dims = []           # store dimension (# of states) of the variables
  i = 0               # (local index over variables)
  cliques = []        # cliques (scopes) of the factors we read in
  factors = []        # the factors themselves

  gen = readFileByTokens(filename,'(),')   # get token generator for the UAI file
  type = next(gen)                   # read file type = Bayes,Markov,Sparse,etc
  nVar = int(next(gen))              # get the number of variables
  for i in range(nVar):              #   and their dimensions (states)
    dims.append( int(next(gen)) )
  nChance = int(next(gen))          # get the number of chance node factors
  nDecision = int(next(gen))        # get the number of decision variables (cliques)
  nUtil   = int(next(gen))          # get the number of utility functions
  for c in range(nChance+nDecision+nUtil):  #   get each clique's scopes
    cSize = int(next(gen))           #   (size of clique)
    cliques.append([])
    for i in range(cSize):           #   ( + list of variable ids)
      v = int(next(gen))
      cliques[-1].append( Var(v,dims[v]) )
    #print cliques[-1]
  chance=cliques[:nChance]
  decision=cliques[nChance:nChance+nDecision]
  util=cliques[nChance+nDecision:]
  cliques = chance + util            # now update to only read chance & util f'ns
  for c in range(nChance+nUtil):          # now read in the factor tables:
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

  return factors[:nChance],decision,factors[nChance:]


def LimidCRA2MMAP(C,D,U):
  """Convert CRA LIMID factors into MMAP factors & query variable list  (Not Implemented)

  Example:

  >>> factors,query = LimidCRA2MMAP(C,D,U) 

  See also readLimidCRA().
  """
  X={}
  Q=[]
  DD=[]
  nC,nD,nU = len(C),len(D),len(U)
  for c in D:
    for v in c:
      X[v.label]=v
  for f in C+U:
    for v in f.vars:
      X[v.label]=v
  nxt = 1+max(list(X.keys())) 
  for d in range(len(D)):        # now run through each decision & create MMAP policy vars
    dd = D[d][-1]                # ??? Last variable is decision ID?
    if not dd in X: continue     # No probability or utility depends on the decision?
    par = VarSet([X[v] for v in D[d][:-1]]) 
    #print("Processing ",dd," parents ",par)
    if par.nrStates()==1:   # no parents => decision variable = map variable
      #DD.append( D[d] )     # don't need to keep factor; do mark variable in query
      Q.append(dd)
      continue              # otherwise, create one map var per config, for that policy
    for p in range(par.nrStates()):
      X[nxt] = Var(nxt,X[dd].states)    # create MAP var for policy
      #print("  new query var ",nxt)
      cliq = D[d][:-1]  
      cliq.extend( [X[dd],X[nxt]] )
      tab = np.ones( (par.nrStates(),) + (X[dd].states,X[dd].states) )
      tab[p,:,:] = np.eye(X[dd].states)
      tab = tab.reshape( tuple(v.states for v in cliq) ) 
      tab = np.squeeze(tab)
      DD.append( Factor(VarSet(cliq), np.transpose(tab, tuple(np.argsort([v.label for v in cliq]))) ) )
      Q.append(X[nxt])
      nxt += 1
  z = Var(nxt, nU)
  fZ = Factor(z, 1.)
  for u in range(nU):
    tmp = U[u]
    newarr = np.ones( (tmp.vars.nrStates(),nU) )
    newarr[:,u] = tmp.table.ravel()
    U[u] = Factor( [v for v in tmp.vars]+[z], newarr.reshape( tmp.dims()+(nU,) ) )
  return C+DD+U+[fZ], Q



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
  



