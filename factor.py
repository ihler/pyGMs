"""
factor.py

Defines variables, variable sets, and dense factors over discrete variables (tables) for graphical models

Version 0.0.1 (2015-09-28)
(c) 2015 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import numpy as np;
from sortedcontainers import SortedSet as sset;


class Var(object):
  """A basic discrete random variable; a pair, (label,#states) """
  label = []
  states = 0
  def __init__(self, label, states):
    self.label  = label
    self.states = states
  def __repr__(self):
    return "Var ({},{})".format(self.label,self.states) 
  def __str__(self):
    return str(self.label)
  def __lt__(self,that):
    return self.label < int(that) #that.label;   # TODO: add support for comparing to integers / index?
  def __le__(self,that):
    return self.label <= int(that) #that.label;
  def __gt__(self,that):
    return self.label > int(that) #that.label;
  def __ge__(self,that):
    return self.label >= int(that) #that.label;
  def __eq__(self,that):              # Note tests only for equality of variable label, not states
    return self.label == int(that) #that.label;
  def __ne__(self,that):
    return not self.__eq__(that)
  def __hash__(self):
    return hash(self.label)
  def __int__(self):
    return self.label
  def __index__(self):
    return self.label

class VarSet(sset):
  """Container for (sorted) set of variables; the arguments to a factor """
  # TODO: switch to np.array1D pair (ids, states)  (int/uint,uint)?
  #   using __get__ to return Var types
  #   use np.union1d, in1d, etc to manipulate

  def dims(self):
    return tuple(d for d in (v.states for v in self)) if len(self) else (1,)
  def nvar(self): # also size?
    return len(self)
  def nrStates(self):
    s = 1
    for d in (v.states for v in self):
      s *= d
    return s
  def nrStatesDouble(self):
    s = 1.0
    for d in (v.states for v in self):
      s *= d
    return s
  def __repr__(self):
    return "{"+','.join(map(str,self))+'}'
  def __str__(self):
    return "{"+','.join(map(str,self))+'}'
  def ind2sub(self,idx):
    return np.unravel_index(idx,self.dims())  
    #return np.unravel_index(idx,self.dims(),order=orderMethod)  
  def sub2ind(self,sub):
    return np.ravel_multi_index(sub,self.dims())
    #return np.ravel_multi_index(sub,self.dims(),order=orderMethod)  
  # todo: needs set equality comparison?  (inherited from sset?)

# TODO: create generator class for enumerating over tuples(vs)?
# def tuples(vs):
#   idx = 0
#   end = vs.nrStates()
#   while idx < end:
#     yeild vs.ind2sub(idx)    # not very efficient method...
#     idx += 1



class __VarSet(sset):
  """Container for (sorted) set of variables; the arguments to a factor """
  def __init__(listable):
    self.__vars = np.array( listable , dtype=object )
  # TODO: switch to np.array1D pair (ids, states)  (int/uint,uint)?
  #   using __get__ to return Var types
  #   use np.union1d, in1d, etc to manipulate

  @property
  def dims(self):
    """Dimension (cardinality / # of states) of each variable in the set"""
    return tuple(d for d in (v.states for v in self.__vars)) if len(self.__vars) else (1,)

  @property
  def nvar(self):
    """Number of variables in the set"""
    return len(self.__vars)

  @property
  def nrStates(self):
    return reduce( lambda s,v: s*v.states , self.__vars, 1 ) 
  
  @property   
  def nrStatesDouble(self):
    return reduce( lambda s,v: s*v.states , self.__vars, 1.0 ) 

  def __repr__(self):
    return "{"+','.join(map(str,self.__vars))+'}'
  def __str__(self):
    return "{"+','.join(map(str,self.__vars))+'}'
  def ind2sub(self,idx):
    return np.unravel_index(idx,self.dims())  #,order=orderMethod) ?
  def sub2ind(self,sub):
    return np.ravel_multi_index(sub,self.dims()) #,order=orderMethod)?
  # todo: needs set equality comparison?  (inherited from sset?)
  # todo: needs all variable set operators (union, intersect, etc)





orderMethod = 'f'   # TODO ??? backward?
# Notes: column-major (order=F) puts last index sequentially ("big endian"): t[0 0 0], t[0 0 1], t[0 1 0] ...
#        row major (order=C) puts 1st index sequentially ("little endian"): t[0 0 0], t[1 0 0], t[0 1 0], ...

class Factor(object):
  """A basic factor<float> class """

  v = VarSet([])       # internal storage for variable set (VarSet)
  t = np.ndarray([])   # internal storage for table (numpy array)

  def __init__(self,vars=None,vals=1.0):
    """Constructor for factor: f( [vars],[vals] ) creates factor over [vars] with table [vals] """
    # TODO: add user-specified order method for values (order=)
    # TODO: accept out-of-order vars list  (=> permute vals as req'd)
    try:
      self.v = VarSet(vars)                             # try building varset with args
    except TypeError:                                   # if not iterable (e.g. single variable)
      self.v = VarSet()                                 #   try just adding it
      self.v.add(vars)
    assert( self.v.nrStates() > 0)

    self.t = np.ndarray(shape=self.v.dims(),dtype=float,order=orderMethod)
    try:
      self.t[:] = vals                                  # try filling factor with "vals"
    except ValueError:                                  # if it's an incompatible shape,
      self.t = np.array(vals,dtype=float).reshape(self.v.dims(),order=orderMethod) #   try again using reshape


  def __build(self,vs,ndarray):
    """Internal build function from numpy ndarray"""
    self.v = vs
    self.t = ndarray
    return self

  #TODO: def assign(self, F) : set self equal to rhs F, e.g., *this = F

  def copy(self):
    """Copy constructor"""
    f = Factor()
    f.v = self.v.copy()
    f.t = self.t.copy() #order=orderMethod)
    return f


  # cvar is by ref or copy?
  def changeVars(self, vars):
    """Copy factor but change its arguments (scope)
    f = changeVars( F, [X7,X5])  =>  f(X7=a,X5=b) = F(X0=a,X1=b)
    """
    v = VarSet(vars)
    newOrder = map(lambda x:v.index(x), vars)
    return Factor(v, self.t.transpose(newOrder))
    #return NotImplemented

  def __repr__(self):
    """Detailed representation: scope (varset) + table memory location"""
    return 'Factor(%s,[0x%x])'%(self.v,self.t.ctypes.data)

  def __str__(self):
    """Basic string representation: scope (varset)"""
    return 'Factor(%s)'%self.v

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
    return self.v.dims()  # TODO: check (empty? etc)
    #return self.t.shape  # TODO: check (empty? etc)
    # TODO: convert to tuple? here / in varset?

  #@property  # TODO: make property
  def numel(self):
    """Number of elements (size) of the tabular factor"""
    return self.t.size


  ################## METHODS ##########################################
  def __getitem__(self,loc):
    """Accessor: F[x1,x2] = F[sub2ind(x1,x2)] = F(X1=x1,X2=x2)"""
    if isinstance(loc, (tuple, list)):
      return self.t[loc]
    else:
      return self.t[self.v.ind2sub(loc)] 

  def __setitem__(self,loc,val):
    """Assign values of the factor: F[i,j,k] = F[idx] = val if idx=sub2ind(i,j,k)"""
    if isinstance(loc, (tuple, list)):
      self.t[loc] = val
    else:
      self.t[self.v.ind2sub(loc)] = val
      #self.t.flat[loc] = val # uses c-contiguous order...

  value = __getitem__        # def f.value(loc): Alternate name for __getitem__

  def valueMap(self,x):
    """Accessor: F[x[i],x[j]] where i,j = F.vars, i.e, x is a map from variables to their values"""
    if self.nvar == 0: return self.t[0]          # if a scalar f'n, nothing to index
    return self.t[tuple(x[v] for v in self.v)]   # otherwise, find entry of table

  def __float__(self):
    """Convert factor F to scalar float if possible; otherwise ValueError"""
    if (self.nvar == 0): return self.t[0]
    else: raise ValueError("Factor is not a scalar; scope {}".format(self.v))

  # missing comparator functions?
  def isnan(self):
    """Check for NaN (not-a-number) entries in the factor's values; true if any NaN present"""
    return self.isAny( (lambda x: np.isnan(x)) )

  def isfinite(self):
    """Check for infinite (-inf, inf) or NaN values in the factor; false if any present"""
    return not self.isAny( (lambda x: not np.isfinite(x)) )

  def isAny(self,test):
    """Generic check for any entries satisfying lambda-expression "test" in the factor"""
    for x in np.nditer(self.t, op_flags=['readonly']):
      if op(x):
        return True
    return False



  #### UNARY OPERATIONS ####
  def __abs__(self):
    """Return the absolute value of F, e.g., G = F.abs()  =>  G(x) = |F(x)| for all x"""
    return Factor().__build( VarSet(self.v) , np.fabs(self.t) )

  abs = __abs__

  def __neg__(self):
    """Return the negative of F, e.g., G = -F  =>  G(x) = -F(x) for all x"""
    return Factor().__build( VarSet(self.v) , np.negative(self.t) )

  def exp(self):
    """Return the exponential of F, e.g., G = F.exp()  =>  G(x) = exp(F(x)) for all x"""
    return Factor().__build( VarSet(self.v) , np.exp(self.t) )

  def __pow__(self,power):
    """Return F raised to a power, e.g., G = F.power(p)  =>  G(x) = ( F(x) )^p for all x"""
    return Factor().__build( VarSet(self.v) , np.power(self.t,power) )

  power = __pow__

  def log(self):    # just use base?
    """Return the natural log of F, e.g., G = F.log()  =>  G(x) = log( F(x) ) for all x"""
    return Factor().__build( VarSet(self.v) , np.log(self.t) )

  def log2(self):
    """Return the log base 2 of F, e.g., G = F.log2()  =>  G(x) = log2( F(x) ) for all x"""
    return Factor().__build( VarSet(self.v) , np.log2(self.t) )

  def log10(self):
    """Return the log base 10 of F, e.g., G = F.log10()  =>  G(x) = log10( F(x) ) for all x"""
    return Factor().__build( VarSet(self.v) , np.log10(self.t) )



  #### IN-PLACE UNARY OPERATIONS ####
  # always return "self" for chaining:  f.negIP().expIP() = exp(-f(x)) in-place
  def absIP(self):
    """Take the absolute value of F, e.g., F.absIP()  =>  F(x) <- |F(x)|"""
    np.fabs(self.t, out=self.t)
    return self

  def expIP(self):
    """Take the exponential of F, e.g., F.expIP()  =>  F(x) <- exp(F(x))"""
    np.exp(self.t, out=self.t)
    return self

  def powerIP(self,power):
    """Raise F to a power, e.g., F.powerIP(p)  =>  F(x) <- ( F(x) )^p"""
    np.power(self.t, power, out=self.t)
    return self

  __ipow__ = powerIP

  def logIP(self):    # just use base?
    """Take the natural log of F, e.g., F.logIP()  =>  F(x) <- log( F(x) )"""
    np.log(self.t, out=self.t)
    return self

  def log2IP(self):
    """Take the log base 2 of F, e.g., F.log2IP()  =>  F(x) <- log2( F(x) )"""
    np.log2(self.t, out=self.t)
    return self

  def log10IP(self):
    """Take the log base 10 of F, e.g., F.log10IP()  =>  F(x) <- log10( F(x) )"""
    np.log10(self.t, out=self.t)
    return self

  def negIP(self):
    """Take the negation of F, e.g., F.negIP()  =>  F(x) <- (-F(x))"""
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
    """In-place addition, F1 += F2.  Best if F2.vars <= F1.vars"""
    #return self.__opExpand1(that,np.add, out=self)
    return self.__opExpand2(that,np.add, out=self)

  def __sub__(self,that):
    """Subtraction of factors, e.g.,  G(x_1,x_2) = F1(x_1) - F2(x_2)"""
    #return self.__opExpand1(that, np.subtract)
    return self.__opExpand2(that,np.subtract)

  def __rsub__(self,that):
    """Right-subtraction, e.g. G(x) = 3.0 - F(x)"""
    B = that if isinstance(that,Factor) else Factor([],that)
    return B.__opExpand2(self, np.subtract)

  def __isub__(self,that):
    """In-place subtraction, F1 -= F2.  Best if F2.vars <= F1.vars"""
    #return self.__opExpand1(that,np.subtract, out=self)
    return self.__opExpand2(that,np.subtract, out=self)

  def __mul__(self,that):
    """Multiplication of factors, e.g.,  G(x_1,x_2) = F1(x_1) * F2(x_2)"""
    return self.__opExpand2(that, np.multiply)

  def __rmul__(self,that):
    """Right-multiplication, e.g. G(x) = 3.0 * F(x)"""
    return self.__opExpand2(that, np.multiply)

  def __imul__(self,that):
    """In-place multiplication, F1 *= F2.  Best if F2.vars <= F1.vars"""
    #return self.__opExpand1(that,np.multiply, out=self)
    return self.__opExpand2(that,np.multiply, out=self)

  def __div__(self,that):
    """Division of factors, e.g.,  G(x_1,x_2) = F1(x_1) / F2(x_2)"""
    return self.__opExpand2(that, np.divide)

  __truediv__ = __div__

  def __rdiv__(self,that):
    """Right-divide, e.g. G(x) = 3.0 / F(x)"""
    B = that if isinstance(that,Factor) else Factor([],that)
    return B.__opExpand2(self, np.divide) 

  __rtruediv__ = __rdiv__

  def __idiv__(self,that):
    """In-place divide, F1 /= F2.  Best if F2.vars <= F1.vars"""
    #return self.__opExpand1(that,np.divide, out=self)
    return self.__opExpand2(that,np.divide, out=self)

  __itruediv__ = __idiv__

  #### ELIMINATION OPERATIONS ####
  # TODO: check for elim non-iterable & if so, make it a list of itself
  def sum(self, elim=None, out=None):
    """Eliminate via sum on F, e.g., f(x_2) = sum_{x_1} F(x_1,x_2) = F.sum(x[1])"""
    if (elim is None):
      elim = self.v
    return self.__opReduce2(self.v & elim,np.sum, out=out)

  def marginal(self, target, out=None):
    """Compute the marginal of F, e.g., f(x_2) = sum_{x_1} F(x_1,x_2) = F.marginal(x[2])"""
    return self.__opReduce2(self.v - target,np.sum, out=out)

  def sumPower(self, elim=None, power=1.0, out=None):
    """Eliminate via powered sum, e.g., f(x_2) =  \\root^{1/p}{ sum_{x_1} F(x_1,x_2)^p } = F.sumPower(x[1],p)"""
    if (elim is None):
      elim = self.v
    #return (self ** power).sum(elim).powerIP(1.0/power)     # fails if elim = self.v?
    tmp = (self ** power).sum(elim)
    tmp **= (1.0/power)
    return tmp
    #return (self ** power).sum(elim)**(1.0/power)     # TODO: make more efficient?

  def lse(self, elim=None, out=None):
    """Eliminate via log-sum-exp on F, e.g., f(x_2) = log sum_{x_1} exp F(x_1,x_2) = F.lse(x[1])"""
    if (elim is None):
      elim = self.v
    return self.__opReduce3(self.v & elim,np.logaddexp.reduce, out=out)
    #return self.__opReduce1(self.v & elim,np.logaddexp, -np.inf)  # TODO: make more efficient!

  def lsePower(self, elim=None, power=1.0, out=None):
    """Eliminate via powered log-sum-exp, e.g., f(x_2) = 1/p log sum_{x_1} exp F(x_1,x_2)*p = F.lsePower(x[1],p)"""
    if (elim is None):
      elim = self.v
    #return (self*power).lse(elim).__imul__(1.0/power)      # fails if elim = self.v?
    tmp = (self*power).lse(elim)
    tmp *= (1.0/power)
    return tmp
    #return (self*power).lse(elim)*(1.0/power)      # TODO: make more efficient?

  def max(self, elim=None, out=None):
    """Eliminate via max on F, e.g., f(x_2) = max_{x_1} F(x_1,x_2) = F.max(x[1])"""
    if (elim is None):
      elim = self.v
    return self.__opReduce2(self.v & elim,np.max, out=out)

  def maxmarginal(self, target, out=None):
    """Compute the max-marginal of F, e.g., f(x_2) = max_{x_1} F(x_1,x_2) = F.maxmarginal(x[2])"""
    return self.__opReduce2(self.v - target,np.max, out=out)

  def min(self, elim=None, out=None):
    """Eliminate via min on F, e.g., f(x_2) = min_{x_1} F(x_1,x_2) = F.min(x[1])"""
    if (elim is None):
      elim = self.v
    return self.__opReduce2(self.v & elim,np.min, out=out)

  def minmarginal(self, target, out=None):
    """Compute the min-marginal of F, e.g., f(x_2) = min_{x_1} F(x_1,x_2) = F.minmarginal(x[2])"""
    return self.__opReduce2(self.v - target,np.min, out=out)


    # use ufunc.reduceat?  reduce etc seem not good?
    # frompyfunc to make ufunc from python function?
    # use "externalloop" flag?
    #return t.max(axis=None,out=None) # use axis to specific dimensions to eliminate; out for IP version


  #### TUPLE OPERATIONS ####
  def argmax2(self, cvars=None, ctuple=None):
    """Find the argmax of the factor, with partial conditioning (as var list + value list) if desired.
    Returns a maximizing configuration of f(x|xc=Xc) as a tuple of states
    """
    if (cvars is None):
      return self.v.ind2sub(self.t.argmax())
    ax = tuple(map(lambda x:ctuple[cvars.index(x)] if  x in cvars else slice(None) ,self.v))
    return self.v.ind2sub(self.t[ax].argmax())

  def argmax(self, evidence={}):
    """Find the argmax of the factor, with partial conditioning (as dict evidence[v]) if desired
    Returns a maximizing configuration of f(x|xc=Xc) as a tuple of states
    """
    if len(evidence)==0:
      return self.v.ind2sub(self.t.argmax())
    ax = tuple([ evidence[v] if v in evidence else slice(None) for v in self.v ])
    return self.v.ind2sub( self.t[ax].argmax() )

  def argmin2(self, cvars=None, ctuple=None):
    """Find the argmin of the factor, with partial conditioning if desired (list+list version)"""
    if (cvars is None):
      return self.v.ind2sub(self.t.argmin())
    ax = tuple(map(lambda x:ctuple[cvars.index(x)] if  x in cvars else slice(None) ,self.v))
    return self.v.ind2sub(self.t[ax].argmin())

  def argmin(self, evidence={}):
    """Find the argmin of the factor, with partial conditioning if desired (dict version)"""
    if len(evidence)==0:
      return self.v.ind2sub(self.t.argmax())
    ax = tuple([ evidence[v] if v in evidence else slice(None) for v in self.v ])
    return self.v.ind2sub( self.t[ax].argmax() )

  def sample(self, Z=None):
    """Draw a random sample (as a tuple of states) from the factor (assumes positive)
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

  def condition2(self, cvars=None,ctuple=None):
    """Create a clamped (or "sliced") factor using partial conditioning (list+list version)
       e.g., F.condition([0,2],[a,b]) = f(x1,x3) = F(x_0=a,x_1,x_2=b,x_3)
    """
    cvars = cvars if (cvars is not None) else VarSet()
    ax = tuple(map(lambda x:ctuple[cvars.index(x)] if  x in cvars else slice(None) ,self.v))
    return Factor(self.v - cvars, self.t[ax])   # forces table copy in constructor

  def condition(self, evidence):
    """Create a clamped (or "sliced") factor using partial conditioning (dict version)
       e.g., F.condition({0:a,2:b}) = f(x1,x3) = F(x_0=a,x_1,x_2=b,x_3)
    """
    ax = tuple([ evidence[v] if v in evidence else slice(None) for v in self.v ])
    cvars = [ v for v in self.v if v in evidence ]
    return Factor(self.v - cvars, self.t[ax])   # forces table copy in constructor

  def slice2(self, cvars=None,ctuple=None):
    """Create a clamped (or "sliced") factor using partial conditioning (list+list version)"""
    return self.condition2(cvars,ctuple)

  def slice(self, evidence={}):
    """Create a clamped (or "sliced") factor using partial conditioning (dict version)"""
    return condition(self,evidence)

  def entropy(self):
    """Compute the entropy of the factor (normalizes, assumes positive)"""
    Z = self.sum()
    assert (Z > 0), 'Non-normalizable factor (perhaps log factor?)' # also check for positivity?
    H = 0.0
    for x in np.nditer(self.t, op_flags=['readonly']):
      p = x/Z
      H += 0.0 if p==0 else -p*np.log(p)
    return H

  def norm(self):
    """Compute any of several norm-like functions on F(x)"""
    # TODO: implement
    return

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
    return Factor( VarSet(v) , self.t.copy(order=orderMethod) ).__opUnaryIP(op)
  
  def __opAccumulate(self,r,op):
    for x in np.nditer(self.t, op_flags=['readonly']):
      r = op(r,x)
    return r

# TODO: at least use numpy "broadcast" / "external_loop" etc ; maybe define ufuncs or compile them?
# 
  def __opExpand1(self,that,op, out=None):
    """Internal combination function; brute force application of arbitrary function "op"; slow """
    A = self
    B = that if isinstance(that,Factor) else Factor([],that)
    vall = A.v | B.v
    axA = list(map(lambda x:A.v.index(x) if  x in A.v else -1 ,vall))
    axB = list(map(lambda x:B.v.index(x) if  x in B.v else -1 ,vall))
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
    A = self
    B = that if isinstance(that,Factor) else Factor([],that)
    vall = A.v | B.v
    dA = list(map(lambda x:x.states if  x in A.v else 1 ,vall))
    dB = list(map(lambda x:x.states if  x in B.v else 1 ,vall))
    if ( (out is not None) and (out.v == vall) ):
        f = out                  # if out can be written to directly, do so
    else: 
        f = Factor(vall)         # otherwise, make storage for output function
    op( A.t.reshape(dA,order='A') , B.t.reshape(dB,order='A'), out=f.t )   # TODO: order=A necessary?
    if (out is not None and f is not out):
        out.__build(f.v,f.t)     # if out requested but not used, write f's table into out
        return out
    return f

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
    else:
      if (out is None):
        out = Factor(VarSet(self.v - elim))
      else:
        assert (out.v == VarSet(self.v-elim) ), "Cannot eliminate into an existing factor with incorrect scope"
      ax = tuple(map(lambda x: self.v.index(x), elim))  #
      #t = np.apply_over_axes(op, self.t, ax)    # bad: lots of intermediate memory?
      #t = op(self.t, axis=ax, out=None)               # any better?  TODO
      #return Factor().__build( VarSet(self.v - elim), t )
      op(self.t, axis=ax, out=out.t)               # any better?  TODO
      return out

  def __opReduce3(self, elim, op, out=None):  # assumes elim <= self.v
    """Internal reduce / eliminate function; assumes "op" is a numpy build-in (using a ufunc)
    works with numpy reduce ops that require single axes at a time
    """
    if ((elim is None) or (len(elim)==len(self.v))):
      return op(np.ravel(self.t))
    else:
      if (out is None):
        out = Factor(VarSet(self.v - elim))
      else:
        assert (out.v == VarSet(self.v-elim) ), "Cannot eliminate into an existing factor with incorrect scope"
      ax = tuple(map(lambda x: self.v.index(x), elim))  #
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

