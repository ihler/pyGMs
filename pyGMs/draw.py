from .factor import *
from .graphmodel import *

def drawMarkovGraph(model,**kwargs):
    """Draw a Markov random field using networkx function calls

    Args:
      ``**kwargs``: remaining keyword arguments passed to networkx.draw()

    Example:

    >>> gm.drawMarkovGraph(model, labels={0:'0', ... } )    # keyword args passed to networkx.draw()
    """
    # TODO: fix defaults; specify shape, size etc. consistent with FG version
    import networkx as nx
    import copy
    kwargs = copy.copy(kwargs);  # make a copy of the arguments dict for mutation
    G = nx.Graph()
    G.add_nodes_from( [v.label for v in model.X if v.states > 1] )  # only non-trivial vars
    for f in model.factors:
      for v1 in f.vars:
        for v2 in f.vars:
          if (v1 != v2): G.add_edge(v1.label,v2.label)
    kwargs['var_labels'] = kwargs.get('var_labels',{n:n for n in G.nodes()})
    kwargs['labels'] = kwargs.get('labels', kwargs.get('var_labels',{}) )
    kwargs.pop('var_labels',None);  # remove artificial "var_labels" entry)
    kwargs['edgecolors'] = kwargs.get('edgecolors','k')
    nx.draw(G,**kwargs)
    return G

def drawFactorGraph(model,var_color='w',factor_color=(.2,.2,.8),**kwargs):
    """Draw a factorgraph using networkx function calls

    Args:
      var_color (str, tuple): networkx color descriptor for drawing variable nodes
      factor_color (str, tuple): networkx color for drawing factor nodes
      var_labels (dict): variable id to label string for variable nodes
      factor_labels (dict): factor id to label string for factor nodes
      ``**kwargs``: remaining keyword arguments passed to networkx.draw()

    Example:

    >>> gm.drawFactorGraph(model, var_labels={0:'0', ... } )    # keyword args passed to networkx.draw()
    """
    # TODO: specify var/factor shape,size, position, etc.; return G? silent mode?
    import networkx as nx
    G = nx.Graph()
    vNodes = [v.label for v in model.X if v.states > 1]   # list only non-trivial variables
    fNodes = [-i-1 for i in range(len(model.factors))]    # use negative IDs for factors
    G.add_nodes_from( vNodes )
    G.add_nodes_from( fNodes )
    for i,f in enumerate(model.factors):
      for v1 in f.vars:
        G.add_edge(v1.label,-i-1)

    if not 'pos' in kwargs: kwargs['pos'] = nx.spring_layout(G) # so we can use same positions multiple times...
    kwargs['var_labels']  = kwargs.get('var_labels',{n:n for n in vNodes})
    kwargs['labels'] = kwargs.get('var_labels',{})
    kwargs['edgecolors'] = kwargs.get('edgecolors','k')
    nx.draw_networkx(G, nodelist=vNodes,node_color=var_color,**kwargs)
    kwargs['labels'] = kwargs.get('factor_labels',{})    # TODO: need to transform?
    nx.draw_networkx_nodes(G, nodelist=fNodes,node_color=factor_color,node_shape='s',**kwargs)
    nx.draw_networkx_edges(G,**kwargs)
    return G



def drawBayesNet(model,**kwargs):
    """Draw a Bayesian Network (directed acyclic graph) using networkx function calls

    Args:
      ``**kwargs``: remaining keyword arguments passed to networkx.draw()

    Example:

    >>> gm.drawBayesNet(model, labels={0:'0', ... } )    # keyword args passed to networkx.draw()
    """
    import networkx as nx
    topo_order = bnOrder(model)                              # TODO: allow user-provided order?
    if topo_order is None: raise ValueError('Topo order not found; model is not a Bayes Net?')
    pri = np.zeros((len(topo_order),))-1
    pri[topo_order] = np.arange(len(topo_order))
    G = nx.DiGraph()
    G.add_nodes_from( [v.label for v in model.X if v.states > 1] )  # only non-trivial vars
    for f in model.factors:
      v2label = topo_order[ int(max(pri[v.label] for v in f.vars)) ]
      for v1 in f.vars:
        if (v1.label != v2label): G.add_edge(v1.label,v2label)

    kwargs['var_labels'] = kwargs.get('var_labels',{n:n for n in [v.label for v in model.X]})
    kwargs['labels'] = kwargs.get('labels', kwargs.get('var_labels',{}) )
    kwargs['arrowstyle'] = kwargs.get('arrowstyle','->')
    kwargs['arrowsize'] = kwargs.get('arrowsize',10)
    kwargs['edgecolors'] = kwargs.get('edgecolors','k')
    nx.draw(G,**kwargs)
    return G


def drawLimid(model, C,D,U, **kwargs):
    """Draw a limited-memory influence diagram (limid) using networkx 

    Args:
      ``**kwargs``: remaining keyword arguments passed to networkx.draw()

    Example:

    >>> model.drawLimid(C,D,U, var_labels={0:'0', ... } )    # keyword args passed to networkx.draw()
    """
    import networkx as nx
    decisions = [d[-1] for d in D]                       # list the decision variables
    model = GraphModel( C + [Factor(d,1.) for d in D] )  # get all chance & decision vars, arcs
    chance = [c for c in model.X if c not in decisions]
    util = [-i-1 for i,u in enumerate(U)]
    cpd_edges, util_edges, info_edges = [],[],[]
    topo_order = bnOrder(model)
    if topo_order is None: raise ValueError('Topo order not found; graph is not a Bayes Net?')
    pri = np.zeros((len(topo_order),))-1
    pri[topo_order] = np.arange(len(topo_order))
    G = nx.DiGraph()
    G.add_nodes_from( [v.label for v in model.X if v.states > 1] )  # only non-trivial vars
    G.add_nodes_from( util )                                       # add utility nodes
    for f in model.factors:
      v2label = topo_order[ int(max(pri[v.label] for v in f.vars)) ]
      for v1 in f.vars:
        if (v1.label != v2label):
          G.add_edge(v1.label,v2label)
          if v2label in decisions: info_edges.append((v1.label,v2label))
          else: cpd_edges.append((v1.label,v2label))
    for i,u in enumerate(U):
      for v1 in u.vars:
        G.add_edge(v1.label,-i-1)
        util_edges.append( (v1.label,-i-1) )
    if not 'pos' in kwargs: kwargs['pos'] = nx.spring_layout(G) # so we can use same positions multiple times...
    for d in decisions: nx.draw_networkx_nodes(G, nodelist=[d], node_color=(.7,.7,.9), node_shape='s', **kwargs)
    for c in chance: nx.draw_networkx_nodes(G, nodelist=[c], node_color=(1.,1.,1.), node_shape='o', **kwargs)
    for u in util: nx.draw_networkx_nodes(G, nodelist=[u], node_color=(.7,.9,.7), node_shape='d', **kwargs)
    nx.draw_networkx_edges(G, edgelist=cpd_edges+util_edges, **kwargs)
    if info_edges:
      tmp = nx.draw_networkx_edges(G, edgelist=info_edges, **kwargs)
      for line in tmp: line.set_linestyle('dashed')
    nx.draw_networkx_labels(G, **kwargs)



#def drawJGraph(



