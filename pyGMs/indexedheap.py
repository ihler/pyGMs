

class IndexedHeap(object):
  """Heap object that allows for re-prioritization of items.
  H = IndexedHeap( [(p0,i0), (p1,i1), ... (pN,iN)] )  # construct heap of items i with priorities p
  H.top()    # view highest priority pair (pJ,iJ)
  H.pop()    # return & remove highest prioirity pair
  H.insert( pNew, iNew )    # add item iNew with priority pNew; replaces if item iNew already in heap
  H.erase( iOld )           # remove item iOld (whatever its priority)
  """

  def __repr__(self):
    return "IndexedHeap; {} items".format(len(self.__p))
  def __str__(self):
    return "IndexedHeap; {} items".format(len(self.__p))

  def __init__(self, iterable=[]):
    """Build an indexed heap.  Expects an iterable of pairs, (priority, item)"""
    self.__p   = []
    self.__id  = []
    self.__rev = {}
    if len(iterable) > 0:
      self.__p,self.__id = (list(z) for z in zip(*iterable))
      self.__rev = { I:i+1 for i,I in enumerate(self.__id) }
      for i in reversed(xrange(len(self.__p))): self.__maxHeapify(i+1)
    
  def __len__(self):
    """Number of entries in the heap"""
    return len(self.__p)

  def clear(self):
    """Remove all elements from the heap"""
    self.__p   = []
    self.__id  = []
    self.__rev = {}

  def insert(self, P, R):
    """Add item R with priority P to the heap, replacing priority if R is already in the heap"""
    I = self.__rev.get(R, 0)
    if I != 0:                # if R already exists, just change its priority 
      self.__p[I-1] = P
      self.__maxHeapify(I)
    else:                     # otherwise add it to the end 
      self.__p.append(P)
      self.__id.append(R)
      I = len(self.__p)
      self.__rev[R] = I
    while True:
      parent = I/2            # now walk upward from inserted point & check heap property
      if parent > 0 and self.__p[parent-1] < self.__p[I-1]:
        self.__heapSwap(parent, I)
        I = parent
      else: 
        return                # quit whenever we're in heap order

  def erase(self, R):
    """Remove item R from the heap, assuming it exists"""
    I = self.__rev.get(R,0)
    if I == 0:                # already removed: done
      return         
    elif I == len(self.__p):  # last item is removable without breaking heap:
      self.__rev.pop(R)
      self.__p.pop()
      self.__id.pop()
    else:                     # swap with end of the heap, then push old end down until heapified
      self.__heapSwap(I,len(self.__p))
      self.__rev.pop(R)
      self.__p.pop()
      self.__id.pop()
    return 

  def is_indexed(self,R):
    """Check if item R is in the heap"""
    return self.__rev.get(R,0) > 0

  def by_index(self,R):
    """Return the priority of item R in the heap, assuming it exists; None otherwise"""
    I = self.__rev.get(R,0)
    return self.__p[I-1] if I > 0 else None

  def pop(self):
    """Return & remove the highest priority item in the heap"""
    p,R = self.top()
    self.erase(self.__id[0])
    return p,R

  def top(self):
    """Return but do not remove the highest priority item in the heap"""
    return self.__p[0],self.__id[0]


  def __heapSwap(self,i,j):                    # simple swapping function
    """Swap helper function"""
    self.__p[i-1], self.__p[j-1] = self.__p[j-1], self.__p[i-1]
    I,J = self.__id[i-1], self.__id[j-1]
    self.__id[i-1], self.__id[j-1] = J, I
    self.__rev[I], self.__rev[J] = self.__rev[J], self.__rev[I]

  def __maxHeapify(self,i):
    """Heapify helper function; push down to maintain heap property from i"""
    while True:
      left,right,largest = 2*i,2*i+1,i    # find largest of i, L[i], R[i]
      if left  <= len(self.__p) and self.__p[left-1]  > self.__p[largest-1]: largest=left
      if right <= len(self.__p) and self.__p[right-1] > self.__p[largest-1]: largest=right
      if largest == i: return             # if i's the largest, we're done
      self.__heapSwap(largest,i)          # otherwise, promote the largest by swapping
      i = largest                         # and push down from i's new position


