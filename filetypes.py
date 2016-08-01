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
from .factor import *


#def OLDreadFileByTokens(path, specials=[]):
#  """
#  Helper function for file IO
#  Iterator to read from file byte by byte, splitting into tokens at spaces or elements of "specials"
#  """
#  #TODO: add support for comment / ignore-content format?  (/* comment */, # comment, etc)
#  with open(path, 'r') as fp:
#    buf = []
#    while True:
#      ch = fp.read(1)
#      if ch == '':
#        break
#      elif ch in specials:
#        if buf:
#          yield ''.join(buf)
#        buf = []
#        yield ch
#      elif ch.isspace():
#        if buf:
#          yield ''.join(buf)
#          buf = []
#      else:
#        buf.append(ch)


def readFileByTokens(path, specials=[]):
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


#class FileTokenizer:
#  """Helper function for file IO"""
#  def __init__(self, path):
#    self.name = path
#    self.fh = open(path)
#    self.eof = False
#  def __enter__(self):
#    return self
#  def __exit__(self,type,value,traceback):
#    self.close()
#  def close(self):
#    self.fh.close()
#    self.eof = True
#  def next(self):
#    buf = []
#    while not self.eof:
#      ch = self.fh.read(1)
#      if (ch == ''):
#        self.eof=True
#        break
#      if (ch.isspace() and buf):
#        break
#      else:
#        buf.append(ch)
#    return ''.join(buf)
#
            
"""
with FileTokenizer('tst.txt') as tok:
  while not tok.eof:
    print tok.next()
"""



def readUai(filename):
  """Read in a collection (list) of factors specified in UAI 2010 format
  factor_list = readUai10( 'path/filename.uai' )
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
    pi = list(map(lambda x:vs.index(x), cliques[c])) 
    ipi = list(pi)                   # get permutation mapping: file's order to sorted order
    for j in range(len(pi)):         #   (ipi = inverse permutation)
      ipi[pi[j]] = j
    #print 'Building %s : %s,%s : %s'%(cliques[c],factorSize,vs,tSize)
    #
    tab = np.array([next(gen) for tup in range(tSize)],dtype=float,order='C').reshape(factorSize)
    t2  = np.transpose(tab, tuple(np.argsort([v.label for v in cliques[c]])))
    factors[-1].table = np.array(t2,dtype=float,order='F')
    #
    if False: # for tup in np.ndindex(factorSize):  # automatically uai order? ("big endian")
      tok = next(gen)
      #print "%s => %s: %s"%(tup,tuple(tup[ipi[j]] for j in range(len(ipi))),tok)
      if (tok == '('):               # check for "sparse" (run-length) representation
        run, comma, val, endparen = next(gen), next(gen), next(gen), next(gen)
        assert(comma == ',' and endparen==')')
        for r in range(run):         #   if so, fill run of table with value
          mytup = tuple(tup[ipi[j]] for j in range(len(ipi)))
          factors[-1][mytup] = float(val)
      else:                          # otherwise just a list of values in the table
        mytup = tuple(tup[ipi[j]] for j in range(len(ipi)))
        factors[-1][mytup] = float(tok)

  # TODO: read evidence if not None (default filename if "True"?)
  # condition on evidence?  Or return tuple?
  # return graphmodel object? 
  return factors

def readEvidence10(evidence_file):
  """Read UAI-2010 evidence file\
     The 2010 specification allowed multiple evidence configurations in the same file.
     evList = readEvidence10('path/file.uai.evid') returns a list of evidence configurations
     evList[i] is a dictionary, { Xi : xi , ... } indicating that variable Xi = value xi.
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
  """Read UAI-2014 evidence file\
     The 2014 specification allowed only one evidence configuration per file.
     ev = readEvidence14('path/file.uai.evid') returns an evidence configuration
     as a dictionary, { Xi : xi , ... } indicating that variable Xi = value xi.
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
      i#fp.write("{:d} ".format(f.nvar) + " ".join(map(str,f.vars[::-1])) + "\n")
      fp.write("{:d} ".format(f.nvar) + " ".join(map(str,f.vars)) + "\n")
    fp.write("\n")                     # (extra line)
    for f in factors:                  # factor tables
      #fp.write("{:d} ".format(f.numel()) + " ".join(map(str,f.t.ravel(order=orderMethod))) + "\n")
      fp.write("{:d} ".format(f.numel()) + " ".join(map(str,f.t.ravel(order='C'))) + "\n")

 



# TODO: test
def readErgo(filename):
  """Read in a Bayesian network (list of conditional probabilities) specified in ERGO format
  factor_list,names,labels = readErgo( 'path/filename.erg' )
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
    pi = list(map(lambda x:vs.index(x), cliques[c]))
    ipi = list(pi)                   # get permutation mapping: file's order to sorted order
    for j in range(len(pi)):         #   (ipi = inverse permutation)
      ipi[pi[j]] = j
    #print 'Building %s : %s,%s : %s'%(cliques[c],factorSize,vs,tSize)
    for tup in np.ndindex(factorSize):  # automatically uai order? ("big endian")
      tok = next(gen)
      #print "%s => %s: %s"%(tup,tuple(tup[ipi[j]] for j in range(len(ipi))),tok)
      if (tok == '('):               # check for "sparse" (run-length) representation
        run, comma, val, endparen = next(gen), next(gen), next(gen), next(gen)
        assert(comma == ',' and endparen==')')
        for r in range(run):         #   if so, fill run of table with value
          mytup = tuple(tup[ipi[j]] for j in range(len(ipi)))
          factors[-1][mytup] = float(val)
      else:                          # otherwise just a list of values in the table
        mytup = tuple(tup[ipi[j]] for j in range(len(ipi)))
        factors[-1][mytup] = float(tok)

  names,labels = [],[]

  for i in range(nVar):
    names.append( str(next(gen)) )

  for i in range(nVar):
    labels.append( [] )
    for j in range(dims[i]):
      labels[-1].append( str(next(gen)) )
  

  # TODO: read evidence if not None (default filename if "True"?)
  # condition on evidence?  Or return tuple?
  # return graphmodel object? 
  return factors,names,labels





# TODO: test
def readWCSP(filename):
  """Read in a weighted CSP (list of neg-log factors) specified in WCSP format
  factor_list,name,upperbound = readWCSP( 'path/filename.wcsp' )
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
  """Write 'filename' in weighted CSP format (see e.g. http://graphmod.ics.uci.edu/group/WCSP_file_format)
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

 


def readOrder(filename):
    """Read an elimination order from a file; format "[nvar] [v0] [v1] ... [vn]
       (note: can also be used for MPE configurations, etc.) """
    with open(filename,'r') as fp:
        lines = fp.readlines();
    text = lines[-1].strip('\n').split(' ');
    nvar = int(text[0]);
    vals = [int(text[i]) for i in range(1,nvar+1)];
    if len(vals) != nvar: raise ValueError("Problem with file?");
    return vals

def writeOrder(filename,order):
    """Write an elimination order (or other vector) to a file"""
    with open(filename,'w') as fp:
        fp.write("{} ".format(len(order)));
        fp.write(" ".join(map(str,order)));
  



