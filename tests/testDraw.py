

import numpy as np
import pyGMs as gm
import matplotlib.pyplot as plt   # use matplotlib for plotting with inline plots
import networkx as nx

J,M,A,E,B = tuple(gm.Var(i,2) for i in range(0,5))  # all binary variables
X = [J,M,A,E,B]   # we'll often refer to variables as Xi or X[i]

# sometimes it's useful to have a reverse look-up from ID to "name string" (e.g. when drawing the graph)
IDtoName = dict( (eval(n).label,n) for n in ['J','M','A','E','B'])


pE = gm.Factor([E], [.998, .002]) # probability of earthquake (false,true)

pB = gm.Factor([B], [.999, .001]) # probability of burglary

pAgEB = gm.Factor([A,E,B], 0.0)   
# Set A,E,B                       # Note: it's important to refer to tuples like
pAgEB[:,0,0] = [.999, .001]       #  (A,E,B)=(0,0,0) in the order of the variables' ID numbers
pAgEB[:,1,0] = [.710, .290]       #  So, since A=X[2], E=X[3], etc., A,E,B is the correct order
pAgEB[:,0,1] = [.060, .940]
pAgEB[:,1,1] = [.050, .950]       # ":" refers to an entire row of the table

pJgA = gm.Factor([J,A], 0.0)      # Probability that John calls given the alarm's status
pJgA[:,0]    = [.95, .05]
pJgA[:,1]    = [.10, .90]

pMgA = gm.Factor([M,A], 0.0)      # Probability that Mary calls given the alarm's status
pMgA[:,0]    = [.99, .01]
pMgA[:,1]    = [.30, .70]

#factors = [pE, pB, pAgEB, pJgA, pMgA]  # collect all the factors that define the model
factors = [pJgA, pMgA, pAgEB, pE, pB]  # collect all the factors that define the model

model = gm.GraphModel([pE,pB,pAgEB,pJgA,pMgA])

bn = gm.GraphModel(factors)       # graphical model object has some useful functions

############# networkx objects

nxg = gm.nxMarkovGraph(bn)
plt.figure();
nx.draw(nxg,labels=IDtoName)
plt.show()


plt.figure()
ax = plt.gca()
gm.drawBayesNet(bn,node_color='w',ax=ax, var_labels=IDtoName);
plt.show()

plt.figure()
ax = plt.gca()
# Weird, now "white" nodes have no boundary?
gm.drawFactorGraph(bn,var_color='w',factor_color=[(.2,.2,.8)],ax=ax, var_labels=IDtoName);
plt.show()

