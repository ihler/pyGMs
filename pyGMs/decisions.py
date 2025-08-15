#from constants import *
# from sortedcontainers import SortedSet;
from pyGMs.factor import Factor
import numpy as np

ZERO = 0
TOL = 1e-8
ONE = np.float64(1.0)
np.seterr(all='ignore') # ignore warn raise


#def factor_to_valuation(factor, factor_type, const_shift=False):
#    if const_shift:
#        # (P, P) = (P,0)*(1,1), (1, U+1) = (1,U)*(1,1) shift valuation by (1,1) adding utility by 1
#        return euFactor(factor.copy(), factor.copy()) if factor_type =='P' else euFactor(Factor(factor.vars.copy(), 1.0), factor.copy()+Factor(factor.vars.copy(), 1.0))
#
#    else:
#        # (P,0) := (P, 1e-300), (1,U)
#        return euFactor(factor.copy(), Factor(factor.vars.copy(), ZERO)) if factor_type == 'P' else euFactor(Factor(factor.vars.copy(), 1.0), factor.copy())

########################################################################################################################
# local helper for euFactor class
########################################################################################################################
def exp_v(term):
    if type(term) in {euFactor, Factor}:
        return term.exp()
    else:
        return np.exp(term)


def log_v(term):
    if type(term) in {euFactor, Factor}:
        return term.log()
    else:
        # if debug and term == 0:
        #     return np.log(ZERO)
        # else:
        #     return np.log(term)
        return np.log(term)


def abs_v(term):
    if type(term) in {euFactor, Factor}:
        return term.abs()
    else:
        return np.abs(term)


def sign_v(term):
    if type(term) in {Factor}:
        return Factor(term.vars, np.sign(term.table)) # term.sign()
    else:
        return np.sign(term)

########################################################################################################################
class euFactor(object):
    """Expected Utility Factor: a factor containing both a probabilistic (f.prob) and utility component (f.util)"""
    def __init__(self, prob=None, util=None):
        """Create an expected utility function. If prob or util are None, extends as (P,0) or (1,U)."""
        if prob is None: prob = 1.
        if util is None: util = 0.
        if not isinstance(prob,Factor): prob = Factor([],prob)
        if not isinstance(util,Factor): util = Factor([],util)
        self.prob = prob
        self.util = util

    def __repr__(self):
        return repr((repr(self.prob), repr(self.util)))

    def __str__(self):
        return str(('P:' + str(self.prob), 'U:' + str(self.util)))

    @property
    def t(self):
        return str(self)

    @property
    def vars(self):
        return self.prob.v | self.util.v

    @property
    def nvar(self):
        return len(self.prob.v | self.util.v)

    # @property
    # def u_div_p_from_log(self):
    #     A = exp_v(self.util)
    #     B = exp_v(self.prob)
    #     C = A/B
    #     if isinstance(C, Factor):
    #         C.t = np.clip(C.t, a_min=-np.exp(EXPLIM), a_max=np.exp(EXPLIM))
    #     else:
    #         if C < -np.exp(EXPLIM):
    #             C = -np.exp(EXPLIM)
    #         elif C > np.exp(EXPLIM):
    #             C = np.exp(EXPLIM)
    #     # return exp_v(self.util) / exp_v(self.prob)
    #     return C
    # @property
    # def u_div_p(self):
    #     return self.util / self.prob

    def copy(self):
        return euFactor(self.prob.copy(), self.util.copy() )

    def mul(self, other):
        # return euFactor(self.prob*other.prob, self.prob*other.util + self.util*other.prob)
        if not hasattr(other,'prob'):  # ATI: interpret scalars as probability terms
          if not hasattr(other,'vars'): other = Factor([],other)
          other = euFactor(other,other)  
        #if not hasattr(other,'prob'): other = euFactor(other,other)  # ATI: interpret scalars as probability terms
        new_p = exp_v(log_v(self.prob) +log_v(other.prob))
        # new_eu = other.util.sign()*exp_v(self.prob.log() + other.util.abs().log()) + self.util.sign()*exp_v(self.util.abs().log() + other.prob.log())
        new_eu = sign_v(other.util) * exp_v( log_v(self.prob) + log_v( abs_v(other.util))) + sign_v(self.util) * exp_v( log_v(abs_v(self.util)) + log_v(other.prob))
        return euFactor(new_p, new_eu)

    def __mul__(self, other):  return self.mul(other) # V1 * V2 --> combination in linear scale
    def __rmul__(self, other): return self.mul(other)
    def __imul__(self, other): tmp = self.mul(other); self.prob,self.util = tmp.prob,tmp.util; return self

    def div(self, other):
        # return euFactor(self.prob/other.prob, self.util/other.prob - self.prob*other.util/other.prob.power(2))
        if not hasattr(other,'prob'): other = euFactor(other,other)  # ATI: interpret scalars as probability terms
        new_p = exp_v(log_v(self.prob) - log_v(other.prob))
        new_eu = sign_v(self.util) * exp_v(log_v(abs_v(self.util)) - log_v(other.prob)) - sign_v(other.util) * exp_v(log_v(self.prob) + log_v(abs_v(other.util)) - 2 * log_v(other.prob))
        return euFactor(new_p, new_eu)

    def __div__(self, other):  return self.div(other) # V1 / V2 = V1 * inv(V2) --> division in linear scale

    # def mul_log(self, other):
    #     # combination in log scale
    #     return euFactor(self.prob+other.prob, log_v(exp_v(self.prob + other.util)+exp_v(other.prob+self.util)))
    #
    # def __add__(self, other):
    #     return self.mul_log(other) # V1 + V2 --> combination of euFactor in log scale

    # def div_log(self, other):
    #     # division in log scale
    #     # if debug: # check if expected utility >= 0
    #     #     check = exp_v(self.util - other.prob) - exp_v(self.prob + other.util - 2 * other.prob)
    #     #     assert np.all(check.t >= 0.0)
    #     return euFactor(self.prob - other.prob, log_v( exp_v(self.util - other.prob) - exp_v(self.prob + other.util - 2 * other.prob)))
    #
    # def __sub__(self, other):
    #     return self.div_log(other) # V1 - V2 = V1 * inv(V2) --> division in log scale

    def logIP(self): # in place transformation to log
        self.prob.logIP()
        # self.prob.t = np.clip(self.prob.t, a_min=-EXPLIM, a_max=EXPLIM)
        self.util.logIP()
        # self.util.t = np.clip(self.util.t, a_min=-EXPLIM, a_max=EXPLIM)

    def expIP(self): # in place transformation to linear
        self.prob.expIP()
        # self.prob.t = np.clip(self.prob.t, a_min=-np.exp(EXPLIM), a_max=np.exp(EXPLIM))
        self.util.expIP()
        # self.util.t = np.clip(self.util.t, a_min=-np.exp(EXPLIM), a_max=np.exp(EXPLIM))

    def exp(self): # create a new valuation in linear scale
        return euFactor(self.prob.exp() if type(self.prob) is Factor else np.exp(self.prob), self.util.exp() if type(self.util) is Factor else np.exp(self.util))

    def log(self): # create a new valuation in log scale
        return euFactor(self.prob.log() if type(self.prob) is Factor else np.log(self.prob), self.util.log() if type(self.util) is Factor else np.log(self.util))

    def abs(self):
        return euFactor(self.prob, self.util.abs())

    def clip_eu_IP(self):
        np.clip(self.util.t, a_min=ZERO, out=self.util.t)

    def max(self, elim=None, out=None):
        return euFactor(self.prob.max(elim, out), self.util.max(elim, out))

    def sum(self, elim=None, out=None):
        return euFactor(self.prob.sum(elim, out), self.util.sum(elim, out))

    def lse(self, elim=None, out=None):
        return euFactor(self.prob.lse(elim, out), self.util.lse(elim, out))

    def min(self, elim=None, out=None):
        return euFactor(self.prob.min(elim, out), self.util.min(elim, out))

    def sumPower(self, elim=None, power=1.0, out=None):
        return euFactor(self.prob.sumPower(elim, power, out), self.util.sumPower(elim, power, out))

    def lsePower(self, elim=None, power=1.0, out=None):
        return euFactor(self.prob.lsePower(elim, power, out), self.util.lsePower(elim, power, out))

    def marginal(self, target, out=None):
        return euFactor(self.prob.marginal(target, out), self.util.marginal(target, out))

    def maxmarginal(self, target, out=None):
        return euFactor(self.prob.maxmarginal(target, out), self.util.maxmarginal(target, out))

    def minmarginal(self, target, out=None):
        return euFactor(self.prob.minmarginal(target, out), self.util.minmarginal(target, out))
