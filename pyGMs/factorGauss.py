"""
factorGauss.py

Defines variables, variable sets, and factors over jointly Gaussian variables for graphical models

Version 0.3.1 (2025-08-15)
(c) 2015-2025 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import numpy as np;
from sortedcontainers import SortedSet as sset;
from .factor import Var,VarSet

class FactorGauss:
  """A basic Gaussian factor class """

  v = VarSet([])       # internal storage for variable set (VarSet)
  # Gaussian, information form: f(x) = exp( -0.5 * x*J*x + x*h' - c )
  h = np.array([],dtype=float)   # information mean vector
  J = np.array([],dtype=float)   # precision / inverse covariance matrix
  c = 0.0                        # constant term

  def __init__(self,vars=None,h=0.0,J=0.0,c=0.0):
    """Constructor for factor: f( [vars],h,J,c ) creates Gaussian factor over [vars] """
    # TODO: add user-specified order method for values (order=)
    # TODO: accept out-of-order vars list  (=> permute vals as req'd)
    try:
      self.v = VarSet(vars)                             # try building varset with args
    except TypeError:                                   # if not iterable (e.g. single variable)
      self.v = VarSet()                                 #   try just adding it
      self.v.add(vars)

    nvar = len(self.v)
    self.h = np.zeros((nvar,))
    self.J = np.zeros((nvar,nvar))
    self.h[:] = h
    self.J[:,:] = J
    self.c = c


  def __build(self,vs,h,J,c):
    """Internal build function from numpy ndarray"""
    self.v = vs
    self.h = h
    self.J = J
    self.c = c
    return self

  #TODO: def assign(self, F) : set self equal to rhs F, e.g., *this = F

  def copy(self):
    """Copy constructor"""
    f = FactorGauss()
    f.v = self.v.copy()
    f.h = self.h.copy()
    f.J = self.J.copy()
    f.c = c
    return f


  # cvar is by ref or copy?
  def changeVars(self, vars):
    """Copy factor but change its arguments (scope)
    f = changeVars( F, [X7,X5])  =>  f(X7=a,X5=b) = F(X0=a,X1=b)
    """
    return NotImplemented

  def __repr__(self):
    """Detailed representation: scope (varset) + table memory location"""
    return 'FactorGauss(%s,[0x%x],[0x%x])'%(self.v,self.h.ctypes.data, self.J.ctypes.data)

  def __str__(self):
    """Basic string representation: scope (varset)"""
    return 'FactorGauss(%s)'%self.v

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

#  #@property
#  def dims(self):
#    """Dimensions (table shape) of the tabular factor"""
#    return self.v.dims()  # TODO: check (empty? etc)
#    #return self.t.shape  # TODO: check (empty? etc)
#    # TODO: convert to tuple? here / in varset?


  ################## METHODS ##########################################
  def __getitem__(self,loc):
    """Accessor: F[x1,x2] = F(X1=x1,X2=x2)"""
    return NotImplemented
    # TODO:  evaluate Gaussian.  Use condition function?

  def __setitem__(self,loc,val):
    """Assign values of the factor: F[i,j,k] = F[idx] = val if idx=sub2ind(i,j,k)"""
    return NotImplemented
    # not implementable -- cannot set f(x) for arbitary x

  value = __getitem__        # def f.value(loc): Alternate name for __getitem__

  def valueMap(self,x):
    """Accessor: F[x[i],x[j]] where i,j = F.vars, i.e, x is a map from variables to their values"""
    if self.nvar == 0: return np.exp(-self.c)          # if a scalar f'n, nothing to index
    return self.t[tuple(x[v] for v in self.v)]   # otherwise, find entry of table

  def __float__(self):
    """Convert factor F to scalar float if possible; otherwise ValueError"""
    if (self.nvar == 0): return np.exp(-self.c)
    else: raise ValueError("Factor is not a scalar; scope {}".format(self.v))



  #### UNARY OPERATIONS ####
  def __pow__(self,power):
    """Return F raised to a power, e.g., G = F.power(p)  =>  G(x) = ( F(x) )^p for all x"""
    return FactorGauss(self.v, self.h * p, self.J * p, self.c * p)

  power = __pow__


  #### IN-PLACE UNARY OPERATIONS ####
  # always return "self" for chaining:  f.negIP().expIP() = exp(-f(x)) in-place

  def powerIP(self,power):
    """Raise F to a power, e.g., F.powerIP(p)  =>  F(x) <- ( F(x) )^p"""
    self.h *= p
    self.J *= p
    self.c *= p
    return self

  __ipow__ = powerIP


  #### BINARY OPERATIONS ####
  # TODO: add boundary cases: 0/0 = ?  inf - inf = ?

  def __mul__(self,that):
    """Multiplication of factors, e.g.,  G(x_1,x_2) = F1(x_1) * F2(x_2)"""
    # if not FactorGauss: cast to float, take log, add to c.
    # else: union vars; h[v1] += h1, J[v1,v1] += J1, h2 J2; c = c1+c2
    if isinstance(that,float) or isinstance(that, int):
      f = FactorGauss(self.v, self.h, self.J, self.c)
      f.c += np.log( float(that) )    # check sign?
    else:
      f = FactorGauss( self.v | that.v )
      i1 = np.array([i for i in range(len(f.v)) if f.v[i] in self.v])
      i2 = np.array([i for i in range(len(f.v)) if f.v[i] in that.v])
      f.c = self.c + that.c
      f.h[i1] += self.h
      f.h[i2] += that.h
      f.J[np.ix_(i1,i1)] += self.J
      f.J[np.ix_(i2,i2)] += that.J
    return f

  def __rmul__(self,that):
    """Right-multiplication, e.g. G(x) = 3.0 * F(x)"""
    return __mul__(self,that)

  def __imul__(self,that):
    """In-place multiplication, F1 *= F2.  Best if F2.vars <= F1.vars"""
    f = self * that
    self.v = f.v
    self.J = f.J
    self.h = f.h
    self.c = f.c
    return self

  def __div__(self,that):
    """Division of factors, e.g.,  G(x_1,x_2) = F1(x_1) / F2(x_2)"""
    # same as mul but subtract that.h, that.J
    if isinstance(that,float) or isinstance(that, int):
      f = FactorGauss(self.v, self.h, self.J, self.c)
      f.c -= np.log( float(that) )    # check sign?
    else:
      f = FactorGauss( self.v | that.v )
      i1 = np.array([i for i in range(len(f.v)) if f.v[i] in self.v])
      i2 = np.array([i for i in range(len(f.v)) if f.v[i] in that.v])
      f.c = self.c - that.c
      f.h[i1] += self.h
      f.h[i2] -= that.h
      f.J[np.ix_(i1,i1)] += self.J
      f.J[np.ix_(i2,i2)] -= that.J
    return f

  __truediv__ = __div__

  def __rdiv__(self,that):
    """Right-divide, e.g. G(x) = 3.0 / F(x)"""
    return __div__( FactorGauss([],c=that) , self )

  __rtruediv__ = __rdiv__

  def __idiv__(self,that):
    """In-place divide, F1 /= F2.  Best if F2.vars <= F1.vars"""
    f = self / that
    self.v = f.v
    self.J = f.J
    self.h = f.h
    self.c = f.c
    return self

  __itruediv__ = __idiv__

  #### ELIMINATION OPERATIONS ####
  # TODO: check for elim non-iterable & if so, make it a list of itself
  def sum(self, elim=None, out=None):
    """Eliminate via sum on F, e.g., f(x_2) = sum_{x_1} F(x_1,x_2) = F.sum(x[1])"""
    if (elim is None):
      elim = self.v
    Sig = np.linalg.inv( self.J )
    mu  = Sig.dot( self.h )  
    keep = np.array([i for i in range(len(self.v)) if self.v[i] not in elim])
    Sig = Sig[np.ix_(keep,keep)]  # extract that submatrix
    mu  = mu[keep]
    # TODO: use out
    f = FactorGauss( self.v - elim )
    f.J = np.linalg.inv(Sig)
    f.h = f.J.dot(mu)
    f.c = self.c  # TODO: ?? not quite right...
    return f

  def marginal(self, target, out=None):
    """Compute the marginal of F, e.g., f(x_2) = sum_{x_1} F(x_1,x_2) = F.marginal(x[2])"""
    return self.sum( self.v - target )

  def sumPower(self, elim=None, power=1.0, out=None):
    """Eliminate via powered sum, e.g., f(x_2) =  \\root^{1/p}{ sum_{x_1} F(x_1,x_2)^p } = F.sumPower(x[1],p)"""
    if (elim is None):
      elim = self.v
    return (self ** power).sum(elim).powerIP(1.0/power)     # TODO: make more efficient?

  def max(self, elim=None, out=None):
    """Eliminate via max on F, e.g., f(x_2) = max_{x_1} F(x_1,x_2) = F.max(x[1])"""
    if (elim is None):
      elim = self.v
    mu = self.mean
    drop = [i for i in range(len(self.v)) if i in elim]
    return self.condition2( elim, mu[drop] )
    # TODO: simpler; directly compute
    # TODO: use out

  def maxmarginal(self, target, out=None):
    """Compute the max-marginal of F, e.g., f(x_2) = max_{x_1} F(x_1,x_2) = F.maxmarginal(x[2])"""
    return self.max( self.v - target, out=out )


  def moments(self):
    """Return moment form (mean, covariance) of the distribution"""
    Sig = np.linalg.inv(self.J)
    mu  = Sig.dot(self.h)
    return mu,Sig

  @property
  def mean(self):
    """the mean of the Gaussian distribution"""
    return self.moments()[0]

  @property
  def cov(self):
    """the covariance of the Gaussian distribution"""
    return np.linalg.inv(self.J)


  #### TUPLE OPERATIONS ####
  def argmax2(self, cvars=None, ctuple=None):
    """Find the argmax of the factor, with partial conditioning (as var list + value list) if desired.
    Returns a maximizing configuration of f(x|xc=Xc) as a tuple of states
    """
    return self.condition2(cvars,ctuple).mean

  def argmax(self, evidence={}):
    """Find the argmax of the factor, with partial conditioning (as dict evidence[v]) if desired
    Returns a maximizing configuration of f(x|xc=Xc) as a tuple of states
    """
    return self.condition(evidence).mean




  def sample(self, Chol=None):
    """Draw a random sample (as a tuple of states) from the factor (assumes positive)
       If option Z=<float> set, the function will assume normalization factor Z
    """
    return NotImplemented
    # TODO: randn( self.h.shape ); if Chol use, else recompute;  return Chol*rand + ...




  def condition2(self, cvars=None,cvals=None):
    """Create a clamped (or "sliced") factor using partial conditioning (list+list version)
       e.g., F.condition2([0,2],[a,b]) = f(x1,x3) = F(x_0=a,x_1,x_2=b,x_3)
    """
    cvars = cvars if (cvars is not None) else VarSet()
    cvals = np.asarray(cvals)
    f = FactorGauss(self.v - cvars)
    keep = [i for i in range(len(self.v)) if self.v[i] not in cvars]
    drop = [i for i in range(len(self.v)) if self.v[i] in cvars]
    f.J[:,:] = self.J[np.ix_(keep,keep)]
    f.h[:] = self.h[keep] - self.J[np.ix_(keep,drop)].dot(cvals)
    f.c = self.c + cvals.dot( self.J[np.ix_(drop,drop)] ).dot(cvals)  # TODO: check shape
    return f

  def condition(self, evidence):
    """Create a clamped (or "sliced") factor using partial conditioning (dict version)
       e.g., F.condition({0:a,2:b}) = f(x1,x3) = F(x_0=a,x_1,x_2=b,x_3)
    """
    return self.condition2( [x for x in evidence], [evidence[x] for x in evidence] )

  def slice2(self, cvars=None,ctuple=None):
    """Create a clamped (or "sliced") factor using partial conditioning (list+list version)"""
    return self.condition2(cvars,ctuple)

  def slice(self, evidence={}):
    """Create a clamped (or "sliced") factor using partial conditioning (dict version)"""
    return self.condition(evidence)


  def entropy(self):
    """Compute the entropy of the factor (normalizes, assumes positive)"""
    return NotImplemented



