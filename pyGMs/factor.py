"""
factor.py

Defines variables, variable sets, and dense factors over discrete variables (tables) for graphical models

Version 0.1.1 (2022-04-06)
(c) 2015-2021 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import numpy as np
#import autograd.numpy as np
from sortedcontainers import SortedSet as sset

## Under testing: cython-compiled variable sets for faster operations
try:
  from pyGMs.varset_c import Var,VarSet
except ImportError:
  #print "Compiled version not loaded; importing python version"
  from pyGMs.varset_py import Var,VarSet    # sortedcontainers version
  #from .varset_py2 import Var,VarSet  # numpy array version



inf = float('inf')


orderMethod = 'F'   # TODO: currently stores in fortran order (as Matlab); should be trivially changable
#orderMethod = 'C'  #   Can we make this "seamless" to the user, and/or force them to do something consistent?

# Notes: column-major (order=F) puts last index sequentially ("big endian"): t[0 0 0], t[0 0 1], t[0 1 0] ...
#        row major (order=C) puts 1st index sequentially ("little endian"): t[0 0 0], t[1 0 0], t[0 1 0], ...

class Factor(object):
  """A basic factor<float> class 

  Factors are the basic building block of our graphical model representations.  In general, a factor
  consists of a set of variables (its "scope"), and a table of values indicating f(x) for each
  joint configuration x (a tuple of values) of its variables.

  Variables are stored in sorted order; most of the time, factors are constructed by reading from files,
  but if built by hand it is safest to use indexing to set the values, e.g.,

  >>> f = Factor( [X,Y,Z], 0.0 )   # builds a factor over X,Y,Z filled with zeros
  >>> f[0,1,0] = 1.5               # set f(X=0,Y=1,Z=0) to 1.5

  Useful attributes are f.vars (the scope) and f.table (the table, a numpy array).

  Factors are imbued with many basic operations for manipulation:
    Operators:  *, +, /, -, **, exp, log, abs, etc.
    In-place versions:  *=, +=, /=, -=, **=, expIP, logIP, etc.
    Elimination: max, min, sum, lse (log-sum-exp), etc.
    Conditioning: return a factor defined by a sub-table, assigning some variables to values
    Other: argmax, argmin, sample, etc., return configurations of X (tuples)

  """

  #v = VarSet([])       # internal storage for variable set (VarSet)
  #t = np.ndarray([])   # internal storage for table (numpy array)

  def __init__(self,vars=VarSet(),vals=1.0):
    """Constructor for Factor class
    
    >>> f = Factor( [X,Y,Z],[vals] )     # creates factor over [X,Y,Z] with table [vals] 

    [vals] should be a correctly sized numpy array, or something that can be cast to the same.
    """
    # TODO: add user-specified order method for values (order=)
    # TODO: accept out-of-order vars list  (=> permute vals as req'd)
    try:
      self.v = VarSet(vars)                             # try building varset with args
    except TypeError:                                   # if not iterable (e.g. single variable)
      self.v = VarSet()                                 #   try just adding it
      self.v.add(vars)
    #assert( self.v.nrStates() > 0)
    #if self.v.nrStatesDouble() > 1e8: raise ValueError("Too big!");

    try: 
      self.t = np.empty(self.v.dims(), float, orderMethod);
      self.t[:] = vals                                  # try filling factor with "vals"
    except ValueError:                                  # if it's an incompatible shape,
      self.t = np.reshape(np.array(vals, float), self.v.dims(), orderMethod)  # try again using reshape

  def __build(self,vs,ndarray):
    """Internal build function from numpy ndarray"""
    self.v = vs
    self.t = ndarray
    return self

  #TODO: def assign(self, F) : set self equal to rhs F, e.g., *this = F

  def copy(self):
    """Copy constructor; make a copy of a factor"""
    return Factor().__build(self.v.copy(),self.t.copy('K'))   # order=orderMethod?

  def changeVars(self, vars, copy=True):
    """Copy a factor but change its arguments (scope).
 
       >>> f = Factor([X0,X1], table)      
       >>> g = changeVars( f, [X7,X5])   # now,  g(X5=b,X7=a) = f(X0=a,X1=b)
    """
    v = VarSet(vars)
    newOrder = map(lambda x:vars.index(x), v)
    if copy: ret = Factor(v, self.t.transpose(newOrder))
    else:    ret = Factor().__build(v, self.t.transpose(newOrder))  # try not to copy if possible
    return ret

  def __repr__(self):
    """Detailed representation: scope (varset) + table memory location"""
    return 'Factor({:s},[0x{:x}])'.format(str(self.v),self.t.ctypes.data)

  def __str__(self):
    """Basic string representation: scope (varset) only"""
    return 'Factor({:s})'.format(str(self.v))

  def latex(self, valueformat="0.4f", factorname="$f(x)$", varnames=None):
    """Return string containing latex code for table values.
    Arguments:
      valueformat : string formatter for values in value column; default "0.4f"
      factorname  : string for header of value column
      varnames    : dict mapping variable ID to string for that column (defaults to $x_i$ if None)
    """
    tex = "\\begin{tabular}[t]{" + "".join(["c" for v in self.v]) + "|c}\n"
    #tex += " & ".join(["$x"+str(int(v))+"$" for v in self.v]) + " & $f_{"+"".join([str(int(v)) for v in self.v])+"}$ \\\\ \\hline \n"
    if varnames is None: varnames = {v:"$x_{"+str(int(v))+"}$" for v in self.v}
    tex += " & ".join([varnames[v] for v in self.v]) + " & "+factorname+" \\\\ \\hline \n"
    for s in range(self.numel()):
        tex += " & ".join([str(si) for si in self.v.ind2sub(s)]) + " & " + ("{:"+valueformat+"}").format(self[s]) + "\\\\ \n"
    tex += "\\end{tabular} \n"
    return tex

  def _repr_markdown_(self, valueformat="0.4f", factorname="$f(x)$", varnames=None):
    """Return iPython markdown code for display
    Arguments:
      valueformat : string formatter for values in value column; default "0.4f"
      factorname  : string for header of value column
      varnames    : dict mapping variable ID to string for that column (defaults to $x_i$ if None)
    """
    md = ""
    if varnames is None: varnames = {v:"X"+str(int(v))+"" for v in self.v}
    #if varnames is None: varnames = {v:"$x_{"+str(int(v))+"}$" for v in self.v}
    md += "| "+" | ".join([varnames[v] for v in self.v]) + " | | "+factorname+" |\n"
    md += "| "+" | ".join([":--:" for i in range(len(self.v)+2)]) + " |\n"
    for s in range(self.numel()):
        md += "| "+" | ".join([str(si) for si in self.v.ind2sub(s)]) + " | | " + ("{:"+valueformat+"}").format(self[s]) + " |\n"
    return md

  @property
  def vars(self):
    """Variables (scope) of the factor; read-only"""
    return self.v
  @vars.setter
  def vars(self,value):
    raise AttributeError("Read-only attribute")

  @property
  def table(self):
    """Table (values, as numpy array) of the factor"""
    return self.t
  @table.setter
  def table(self,values):
    try:
      self.t[:] = values                                # try filling factor with "values"
    except ValueError:                                  # if it's an incompatible shape,
      self.t = np.array(values,dtype=float).reshape(self.v.dims(),order=orderMethod) #   try again using reshape

  @property
  def nvar(self):
    """Number of arguments (variables, scope size) for the factor"""
    return len(self.v)

  #@property
  def dims(self):
    """Dimensions (table shape) of the tabular factor"""
    return self.t.shape  

  #@property  # TODO: make property?
  def numel(self):
    """Number of elements (size) of the tabular factor"""
    return self.t.size


  ################## METHODS ##########################################
  def __getitem__(self,loc):
    """Accessor: F[x1,x2] = F[sub2ind(x1,x2)] = F(X1=x1,X2=x2)"""
    if isinstance(loc, dict): return self.valueMap(loc)
    if self.t.ndim == 1 or isinstance(loc, (tuple, list)):
      return self.t[loc]
    else:
      try:
        return self.t[self.v.ind2sub(loc)] 
      except ValueError:
        raise IndexError("Index {} invalid for table with size {}".format(loc,self.t.shape))

  def __setitem__(self,loc,val):
    """Assign values of the factor: F[i,j,k] = F[idx] = val if idx=sub2ind(i,j,k)"""
    if isinstance(loc, dict): return self.setValueMap(loc,val)
    if self.t.ndim == 1 or isinstance(loc, (tuple, list)):
      self.t[loc] = val
    else:
      try:
        self.t[self.v.ind2sub(loc)] = val
      #self.t.flat[loc] = val # uses c-contiguous order...
      except ValueError:
        raise IndexError("Index {} invalid for table with size {}".format(loc,self.t.shape))

  #value = __getitem__        # def f.value(loc): Alternate name for __getitem__
  def value(self,x):
    """Type-safe version of __getitem__: returns scalar float entry of table at tuple x, or exception"""
    if self.nvar == 0: return self.t[0]
    return self.t.item(x)

  def setValue(self,x,val):
    """Type-safe version of __setitem__: sets a scalar float entry of table at tuple x, or exception"""
    self.t.itemset(x,val)


  def valueMap(self,x):
    """Accessor: F[x[i],x[j]] where i,j = F.vars, i.e, x is a map from variables to their state values"""
    ret = self.t[tuple(x[v] for v in self.v)]
    if self.nvar == 0: return self.t[0]          # if a scalar f'n, nothing to index
    return self.t[tuple(x[v] for v in self.v)]   # otherwise, find entry of table

  def setValueMap(self,x,val):
    """Set F[x[i],x[j]] = val, where i,j = F.vars, i.e, x is a map from variables to their state values"""
    self.t[tuple(x[v] for v in self.v) if len(self.v) else 0] = val  # lookup location to set, or 0 if scalar f'n
 
  def __float__(self):
    """Convert factor F to scalar float if possible; otherwise raises ValueError"""
    if (self.nvar == 0): return self.t[0]
    else: raise ValueError("Factor is not a scalar; scope {}".format(self.v))

  # TODO missing comparator functions?

  def isnan(self):
    """Check for NaN (not-a-number) entries in the factor's values; true if any NaN present"""
    return self.isAny( (lambda x: np.isnan(x)) )

  def isfinite(self):
    """Check for infinite (-inf, inf) or NaN values in the factor; false if any present"""
    return not self.isAny( (lambda x: not np.isfinite(x)) )

  def isAny(self,test):
    """Generic check for any entries satisfying lambda-expression "test" in the factor"""
    for x in np.nditer(self.t, op_flags=['readonly']):
      if test(x):
        return True
    return False



  #### UNARY OPERATIONS ####
  def __abs__(self):
    """Return the absolute value of F:   G = F.abs()  =>  G(x) = | F(x) | for all x"""
    return Factor().__build( self.v.copy() , np.fabs(self.t) ) 

  abs = __abs__

  def __neg__(self):
    """Return the negative of F:         G = -F  =>  G(x) = -F(x) for all x"""
    return Factor().__build( self.v.copy() , np.negative(self.t) )

  def exp(self):
    """Return the exponential of F:      G = F.exp()  =>  G(x) = exp(F(x)) for all x"""
    return Factor().__build( self.v.copy() , np.exp(self.t) )

  def __pow__(self,power):
    """Return F raised to a power:       G = F.power(p)  =>  G(x) = ( F(x) )^p for all x"""
    return Factor().__build( self.v.copy() , np.power(self.t,power) )

  power = __pow__

  def log(self):    # just use base?
    """Return the natural log of F:      G = F.log()  =>  G(x) = log( F(x) ) for all x"""
    with np.errstate(divide='ignore'):
      return Factor().__build( self.v.copy() , np.log(self.t) )

  def log2(self):
    """Return the log base 2 of F:       G = F.log2()  =>  G(x) = log2( F(x) ) for all x"""
    with np.errstate(divide='ignore'):
      return Factor().__build( self.v.copy() , np.log2(self.t) )

  def log10(self):
    """Return the log base 10 of F:      G = F.log10()  =>  G(x) = log10( F(x) ) for all x"""
    with np.errstate(divide='ignore'):
      return Factor().__build( self.v.copy() , np.log10(self.t) )



  #### IN-PLACE UNARY OPERATIONS ####
  # always return "self" for chaining:  f.negIP().expIP() = exp(-f(x)) in-place
  def absIP(self):
    """Take the absolute value of F:     F.absIP()  =>  F(x) <- |F(x)|  (in-place)"""
    np.fabs(self.t, out=self.t)
    return self

  def expIP(self):
    """Take the exponential of F:        F.expIP()  =>  F(x) <- exp(F(x))  (in-place)"""
    np.exp(self.t, out=self.t)
    return self

  def powerIP(self,power):
    """Raise F to a power:               F.powerIP(p)  =>  F(x) <- ( F(x) )^p  (in-place)"""
    np.power(self.t, power, out=self.t)
    return self

  __ipow__ = powerIP

  def logIP(self):    # just use base?
    """Take the natural log of F:       F.logIP()  =>  F(x) <- log( F(x) )  (in-place)"""
    with np.errstate(divide='ignore'):
      np.log(self.t, out=self.t)
    return self

  def log2IP(self):
    """Take the log base 2 of F:        F.log2IP()  =>  F(x) <- log2( F(x) )  (in-place)"""
    with np.errstate(divide='ignore'):
      np.log2(self.t, out=self.t)
    return self

  def log10IP(self):
    """Take the log base 10 of F:       F.log10IP()  =>  F(x) <- log10( F(x) )  (in-place)"""
    with np.errstate(divide='ignore'):
      np.log10(self.t, out=self.t)
    return self

  def negIP(self):
    """Take the negation of F:          F.negIP()  =>  F(x) <- (-F(x))  (in-place)"""
    np.negative(self.t, out=self.t)
    return self



  #### BINARY OPERATIONS ####
  # TODO: add boundary cases: 0/0 = ?  inf - inf = ?
  def __add__(self,that):
    """Addition of factors, e.g.,  G(x_1,x_2) = F1(x_1) + F2(x_2)"""
    return self.__opExpand2(that,np.add)

  def __radd__(self,that):
    """Right-addition, e.g. G(x) = 3.0 + F(x)"""
    return self.__opExpand2(that,np.add)

  def __iadd__(self,that):
    """In-place addition, F1 += F2.  Most efficient if F2.vars <= F1.vars"""
    return self.__opExpand2(that,np.add, out=self)

  def __sub__(self,that):
    """Subtraction of factors, e.g.,  G(x_1,x_2) = F1(x_1) - F2(x_2)"""
    return self.__opExpand2(that,np.subtract)

  def __rsub__(self,that):
    """Right-subtraction, e.g. G(x) = 3.0 - F(x)"""
    B = that if isinstance(that,Factor) else Factor([],that)
    return B.__opExpand2(self, np.subtract)

  def __isub__(self,that):
    """In-place subtraction, F1 -= F2.  Most efficient if F2.vars <= F1.vars"""
    return self.__opExpand2(that,np.subtract, out=self)

  def __mul__(self,that):
    """Multiplication of factors, e.g.,  G(x_1,x_2) = F1(x_1) * F2(x_2)"""
    return self.__opExpand2(that, np.multiply)

  def __rmul__(self,that):
    """Right-multiplication, e.g. G(x) = 3.0 * F(x)"""
    return self.__opExpand2(that, np.multiply)

  def __imul__(self,that):
    """In-place multiplication, F1 *= F2.  Most efficient if F2.vars <= F1.vars"""
    return self.__opExpand2(that,np.multiply, out=self)

  def __div__(self,that):
    """Division of factors, e.g.,  G(x_1,x_2) = F1(x_1) / F2(x_2)"""
    with np.errstate(divide='ignore'):
      return self.__opExpand2(that, np.divide)

  __truediv__ = __div__

  def __rdiv__(self,that):
    """Right-divide, e.g. G(x) = 3.0 / F(x)"""
    B = that if isinstance(that,Factor) else Factor([],that)
    with np.errstate(divide='ignore'):
      return B.__opExpand2(self, np.divide) 

  __rtruediv__ = __rdiv__

  def __idiv__(self,that):
    """In-place divide, F1 /= F2.  Most efficient if F2.vars <= F1.vars"""
    with np.errstate(divide='ignore'):
      return self.__opExpand2(that,np.divide, out=self)

  __itruediv__ = __idiv__

  #### ELIMINATION OPERATIONS ####
  def sum(self, elim=None, out=None):
    """Eliminate via sum on F, e.g., f(X_2) = \sum_{x_1} F(x_1,X_2) = F.sum([X1])"""
    if (elim is None): elim = self.v
    return self.__opReduce2(self.v & elim,np.sum, out=out)

  def marginal(self, target, out=None):
    """Compute the marginal of F, e.g., f(X_2) = \sum_{x_1} F(x_1,X_2) = F.marginal([X2])"""
    return self.__opReduce2(self.v - target,np.sum, out=out)

  def sumPower(self, elim=None, power=1.0, out=None):
    """Eliminate via powered sum, e.g., f(X_2) =  \\root^{1/p}{ sum_{x_1} F(x_1,X_2)^p } = F.sumPower([X1],p)"""
    if (elim is None): elim = self.v
    tmp = (self ** power).sum(elim)
    tmp **= (1.0/power)
    return tmp

  def lse(self, elim=None, out=None):
    """Eliminate via log-sum-exp on F, e.g., f(X_2) = log \sum_{x_1} exp F(x_1,X_2) = F.lse([X1])"""
    if (elim is None): elim = self.v
    return self.__opReduce3(self.v & elim, np.logaddexp.reduce, out=out)

  def lsePower(self, elim=None, power=1.0, out=None):
    """Eliminate via powered log-sum-exp, e.g., f(X_2) = 1/p log \sum_{x_1} exp F(x_1,X_2)*p = F.lsePower([X_1],p)"""
    if (elim is None): elim = self.v
    if   power == inf: return self.max(elim)
    elif power == -inf: return self.min(elim)
    elif power == 1.0:  return self.lse(elim)
    else:
      tmp = (self*power).lse(elim)
      tmp *= (1.0/power)
      return tmp

  def max(self, elim=None, out=None):
    """Eliminate via max on F, e.g., f(X_2) = \max_{x_1} F(x_1,X_2) = F.max([X1])"""
    if (elim is None): elim = self.v
    return self.__opReduce2(self.v & elim,np.max, out=out)

  def maxmarginal(self, target, out=None):
    """Compute the max-marginal of F, e.g., f(X_2) = \max_{x_1} F(x_1,X_2) = F.maxmarginal([X2])"""
    return self.__opReduce2(self.v - target,np.max, out=out)

  def min(self, elim=None, out=None):
    """Eliminate via min on F, e.g., f(X_2) = \min_{x_1} F(x_1,X_2) = F.min([X1])"""
    if (elim is None): elim = self.v
    return self.__opReduce2(self.v & elim,np.min, out=out)

  def minmarginal(self, target, out=None):
    """Compute the min-marginal of F, e.g., f(X_2) = \min_{x_1} F(x_1,X_2) = F.minmarginal([X2])"""
    return self.__opReduce2(self.v - target,np.min, out=out)


    # use ufunc.reduceat?  reduce etc seem not good?
    # frompyfunc to make ufunc from python function?
    # use "externalloop" flag?
    #return t.max(axis=None,out=None) # use axis to specific dimensions to eliminate; out for IP version


  #### TUPLE OPERATIONS ####
  def argmax2(self, cvars=None, ctuple=None):
    """Find the argmax of the factor, with partial conditioning (as var list + value list) if desired.

    Returns a maximizing configuration of f(X|Xc=xc) as a tuple of states
    """
    if (cvars is None):
      return self.v.ind2sub(self.t.argmax())
    ax = tuple(ctuple[cvars.index(x)] if  x in cvars else slice(None) for x in self.v)
    return self.v.ind2sub(self.t[ax].argmax())

  def argmax(self, evidence={}):
    """Find the argmax of the factor, with partial conditioning (as dict evidence[v]) if desired

    Returns a maximizing configuration of f(X|Xc=xc) as a tuple of states
    """
    if len(evidence)==0:
      return self.v.ind2sub(self.t.argmax())
    ax = tuple(evidence[v] if v in evidence else slice(None) for v in self.v)
    return self.v.ind2sub( self.t[ax].argmax() )

  def argmin2(self, cvars=None, ctuple=None):
    """Find the argmin of the factor, with partial conditioning if desired (list+list version)"""
    if (cvars is None):
      return self.v.ind2sub(self.t.argmin())
    ax = tuple(ctuple[cvars.index(x)] if  x in cvars else slice(None) for x in self.v)
    return self.v.ind2sub(self.t[ax].argmin())

  def argmin(self, evidence={}):
    """Find the argmin of the factor, with partial conditioning if desired (dict version)"""
    if len(evidence)==0:
      return self.v.ind2sub(self.t.argmax())
    ax = tuple(evidence[v] if v in evidence else slice(None) for v in self.v)
    return self.v.ind2sub( self.t[ax].argmax() )

  def sample(self, Z=None):
    """Draw a random sample (as a tuple of states) from the factor; assumes positivity.

    If option Z=<float> set, the function will assume normalization factor Z
    """
    Z = Z if Z is not None else self.sum()    # normalize if desired / by default
    assert (Z > 0), 'Non-normalizable factor (perhaps log factor?)' # also check for positivity?
    pSoFar = 0.0
    pDraw = Z * np.random.random_sample()
    it = np.nditer(self.t, op_flags=['readonly'], flags=['multi_index'])  # for tuple return
    #it = np.nditer(self.t, op_flags=['readonly'], flags=[orderMethod+'_index'])  # for index return
    while not it.finished:
      pSoFar += it[0]
      if ( pSoFar  > pDraw ):
        return it.multi_index   # multi_index for tuple return
        #return it.index    # index for index return
      it.iternext()
    return self.v.ind2sub(self.numel()-1)   # if numerical issue: return final state

  def condition2(self, cvars=[],ctuple=[]):
    """Create a clamped (or "sliced") factor using partial conditioning (list+list version)

    >>> F.condition2([0,2],[a,b])   # returns  f(X_1,X_3) = F(X_0=a, X_1, X_2=b, X_3)
    """
    ax = tuple(ctuple[cvars.index(x)] if  x in cvars else slice(None) for x in self.v)
    return Factor(self.v - cvars, self.t[ax])   # forces table copy in constructor

  def condition(self, evidence):
    """Create a clamped (or "sliced") factor using partial conditioning (dict version)

    >>> F.condition({0:a,2:b})   # returns  f(X_1,X_3) = F(X_0=a, X_1, X_2=b, X_3)
    """
    ax = tuple( evidence[v] if v in evidence else slice(None) for v in self.v )
    cvars = [ v for j,v in enumerate(self.v) if ax[j] != slice(None) ] 
    return Factor(self.v - cvars, self.t[ax])   # forces table copy in constructor

  slice2 = condition2   # alternate, libDAI-like names
  slice = condition

  # TODO: assign_slice( evidence, conditional_table ) 
  #    create ax = tuple( ... );  self.table(ax) = conditional_table
  # TODO: assign_slice2( vars, vals, conditional_table )

  def entropy(self):
    """Compute the entropy of the factor (normalizes, assumes positive)"""
    Z = self.sum()
    if not (Z > 0): raise ValueError('Non-normalizable factor (perhaps log factor?)')
    tmp = np.ravel(self.t)
    H = -np.dot( tmp, np.log(tmp.clip(min=1e-300)) )/Z + np.log(Z)  # entropy of tmp/Z
    return H

  def norm(self, distance):
    """Compute any of several norm-like functions on F(x).

    'distance' can be any of:
       'L1'    : L1 or manhattan distance, sum of absolute values
       'L2'    : L2 or Euclidean distance, sum of squares
       'LInf'  : L-Infinity distance, maximum value
       'KL'    : Shannon entropy (KL = Kullback Leibler)
       'HPM'   : Hilbert's projective metric
    """
    distance = distance.lower()
    if   distance == 'l1':   return self.abs().sum()
    elif distance == 'l2':   return (self*self).sum()
    elif distance == 'linf': return self.abs().max()
    elif distance == 'kl':   return self.entropy()
    elif distance == 'hpm':  F = self.log(); return F.max() - F.min();
    else: raise ValueError("Unrecognized norm type {}; 'L1','L2','LInf','KL','HPM'".format(distance));

  def distance(self, that, distance):
    """Compute any of several norm-like functions on F(x).

    'distance' can be any of:
       'L1'    : L1 or manhattan distance, sum of absolute values
       'L2'    : L2 or Euclidean distance, sum of squares
       'LInf'  : L-Infinity distance, maximum value
       'KL'    : Shannon entropy (KL = Kullback Leibler)
       'HPM'   : Hilbert's projective metric
    """
    distance = distance.lower()
    tmp = self.copy()
    if   distance == 'l1':   tmp -= that; tmp.absIP(); return tmp.sum()
    elif distance == 'l2':   tmp -= that; tmp *= tmp;  return tmp.sum()
    elif distance == 'linf': tmp -= that; tmp.absIP(); return tmp.max()
    elif distance == 'kl':   Z=tmp.sum(); tmp/=that; tmp*=that.sum()/Z; tmp.logIP(); tmp*=self; return tmp.sum()/Z;
    elif distance == 'hpm':  tmp /= that; tmp.logIP(); return tmp.max() - tmp.min();
    else: raise ValueError("Unrecognized norm type {}; 'L1','L2','LInf','KL','HPM'".format(distance));
    
    
#useful things:
# np.ndindex(shape) : iterate over tuples consistent with shape
# for index, x in np.ndenumerate(a):  iterate over tuples, values

#def mean(factorList):
# return

#def geomean(factorList):
# return

  ############################ INTERNAL ##############################################
  
  # slow version with arbitrary operator
  def __opUnaryIP(self,op):
    for x in np.nditer(self.t, op_flags=['readwrite']):
      x[...] = op(x)
    return self

  def __opUnary(self,op):
    return Factor( self.v.copy() , self.t.copy(order=orderMethod) ).__opUnaryIP(op)
  
  #def __opAccumulate(self,r,op):
  #  for x in np.nditer(self.t, op_flags=['readonly']):
  #    r = op(r,x)
  #  return r

# TODO: at least use numpy "broadcast" / "external_loop" etc ; maybe define ufuncs or compile them?
# 
  def __opExpand1(self,that,op, out=None):
    """Internal combination function; brute force application of arbitrary function "op"; slow """
    A = self
    B = that if isinstance(that,Factor) else Factor([],that)
    vall = A.v | B.v
    axA = list(A.v.index(x) if  x in A.v else -1 for x in vall)
    axB = list(B.v.index(x) if  x in B.v else -1 for x in vall)
    if ( (not (out is None)) and (out.v == vall) ):
      f = out
    else:
      f = Factor(vall)   # TODO: should also change "out" if specified!
    it = np.nditer([A.t, B.t, f.t], 
      op_axes = [ axA, axB, None ], 
      op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for (i,j,k) in it:
      op(i,j,out=k)
    return f

  def __opExpand2(self,that,op, out=None):
    """Internal combination function; assumes "op" is a numpy build-in (using a ufunc)"""
    if not isinstance(that,Factor): # if not a Factor, must be a scalar; use scalar version:
      if out is None: return Factor(self.v, op(self.t,that)) # with constructor
      else: op(self.t, that, out=out.t); return out          # or direct write
    # Non-scalar 2nd argument version:
    A = self
    B = that if isinstance(that,Factor) else Factor([],that)
    vall = A.v | B.v
    dA,dB = vall.expand_dims(A.v, B.v);
    #dA = tuple(x.states if  x in A.v else 1 for x in vall);
    #dB = tuple(x.states if  x in B.v else 1 for x in vall);
    if ( (out is not None) and (out.v == vall) ): # if out can be written to directly, do so
      op( A.t.reshape(dA,order='A') , B.t.reshape(dB,order='A'), out=out.t )   # TODO: order=A necessary?
    else:
      t = op( A.t.reshape(dA,order='A') , B.t.reshape(dB,order='A') )   # TODO: order=A necessary?
      if (out is None): out = Factor()
      if (len(vall)==0): t = np.asarray([t],dtype=float)
      out.__build(vall,t)
    return out


  def __opReduce1(self,elim,op,init): # TODO: change to IP; caller initializes?
    """Internal reduce / eliminate function; brute force application of arbitrary f'n "op"; slow """
    A = self.t
    f = Factor( self.v - elim , init) # TODO: fill with ???  (0.0 for sum, -inf for lse, etc)
    axA = list(range(len(self.v)))
    axC = list(map(lambda x:f.v.index(x) if  x in f.v else -1 ,self.v))
    C = f.t
    it = np.nditer([A, C], op_axes = [ axA, axC ], flags=['reduce_ok'], op_flags=[['readonly'], ['readwrite']])
    for (i,j) in it:
      op(i,j,out=j)
    return f

  def __opReduce2(self, elim, op, out=None):  # assumes elim <= self.v
    """Internal reduce / eliminate function; assumes "op" is a numpy build-in (using a ufunc)"""
    if ((elim is None) or (len(elim)==len(self.v))):
      return op(self.t)
    elif (out is None):   # non-in-place version
      ax = tuple(self.v.index(x) for x in elim)
      out = Factor(self.v - elim, op(self.t, axis=ax))   
    else:                 # in-place version
      assert (out.v == (self.v-elim) ), "Cannot eliminate into an existing factor with incorrect scope"
      ax = tuple(self.v.index(x) for x in elim)
      op(self.t, axis=ax, out=out.t) 
    return out

  def __opReduce3(self, elim, op, out=None):  # assumes elim <= self.v
    """Internal reduce / eliminate function; assumes "op" is a numpy build-in (using a ufunc)
    works with numpy reduce ops that require single axes at a time
    """
    if ((elim is None) or (len(elim)==len(self.v))):
      return op(np.ravel(self.t))
    else:
      if (out is None):
        out = Factor(self.v - elim)
      else:
        assert (out.v == (self.v-elim) ), "Cannot eliminate into an existing factor with incorrect scope"
      ax = tuple(self.v.index(x) for x in elim) 
      src = self.t
      while len(ax) > 1:
        src = op(src,axis=ax[-1])
        ax = ax[:-1]
      op(src, axis=ax, out=out.t)   
      return out



""" NumPy reduce example:
>>> a = np.arange(24).reshape(2,3,4)
>>> b = np.array(0)
>>> for x, y in np.nditer([a, b], flags=['reduce_ok', 'external_loop'],
...                     op_flags=[['readonly'], ['readwrite']]):
...     y[...] += x
...

Notes
  xhat[ [v.label for v in f.var] ] = list(f.argmax())

"""

