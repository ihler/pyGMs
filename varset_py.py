import numpy as np
from sortedcontainers import SortedSet as sset;
from functools import reduce

class Var(object):
  " ""A basic discrete random variable; a pair, (label,#states) "" "
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
    return self.label < int(that) 
  def __le__(self,that):
    return self.label <= int(that)
  def __gt__(self,that):
    return self.label > int(that) 
  def __ge__(self,that):
    return self.label >= int(that) 
  def __eq__(self,that):              # Note tests only for equality of variable label, not states
    return self.label == int(that) 
  def __ne__(self,that):
    return not self.__eq__(that)
  def __hash__(self):
    return hash(self.label)
  def __int__(self):
    return self.label
  def __index__(self):
    return self.label

class VarSet(sset):
  " ""Container for (sorted) set of variables; the arguments to a factor "" "
  # TODO: switch to np.array1D pair (ids, states)  (int/uint,uint)?
  #   using __get__ to return Var types
  #   use np.union1d, in1d, etc to manipulate

  def dims(self):
    return tuple(v.states for v in self) if len(self) else (1,)
  def nvar(self): # also size?
    return len(self)
  def nrStates(self):
    return reduce( lambda s,v: s*v.states, self, 1);   # TODO: faster? slower?
  def nrStatesDouble(self):
    return reduce( lambda s,v: s*v.states, self, 1.0);
  def __repr__(self):
    return "{"+','.join(map(str,self))+'}'
  def __str__(self):
    return "{"+','.join(map(str,self))+'}'
  def ind2sub(self,idx):
    return np.unravel_index(idx,self.dims())  
    #return np.unravel_index(idx,self.dims(),order=orderMethod)  
  def sub2ind(self,sub):
    return np.ravel_multi_index(sub,self.dims())
  def __hash__(self):
    return hash(tuple(v.label for v in self))
  @property
  def labels(self):
    return [v.label for v in self]
  def expand_dims(self, *iterables):
    return tuple( tuple(map(lambda x:x.states if x in that else 1, self)) for that in iterables);
     #dA = tuple(map(lambda x:x.states if  x in A.v else 1 ,vall));
    #dB = tuple(map(lambda x:x.states if  x in B.v else 1 ,vall));

    #return np.ravel_multi_index(sub,self.dims(),order=orderMethod)  
  # todo: needs set equality comparison?  (inherited from sset?)

