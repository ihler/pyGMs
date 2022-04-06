from .factor import *
from .graphmodel import *


class nxDefaults:
  """Set networkx drawing default appearances for variable nodes, factor nodes, and edges"""
  var = {'edgecolors':'black', 'node_color':'w', 'node_shape':'o' }
  factor = { 'edgecolors':'black', 'node_color':[(.2,.2,.8)], 'node_shape':'s' }
  edge = {'edge_color':'black' }
  util = {'node_color': [(.7,.9,.7)], 'node_shape':'d' }



def nxMarkovGraph(model, all_vars=False):
    """Get networkx Graph object of the Markov graph of the model

    Example:
    >>> G = nxMarkovGraph(model)
    >>> nx.draw(G)
    """
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from( [v.label for v in model.X if (all_vars or v.states > 1)] )
    for f in model.factors:
      for v1 in f.vars:
        for v2 in f.vars:
          if (v1 != v2) and (all_vars or (v1.states > 1 and v2.states > 1)):
            G.add_edge(v1.label,v2.label)
    return G
    """ Plotting examples:
    fig,ax=plt.subplots(1,2)
    pos = nx.spring_layout(G) # so we can use same positions multiple times...
    # use nodelist=[nodes-to-draw] to only show nodes in model
    nx.draw(G,with_labels=True,labels={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6'},
              node_color=[.3,.3,.3,.7,.7,.7,.7],vmin=0.0,vmax=1.0,pos=pos,ax=ax[0])
    nx.draw(G,with_labels=True,labels={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6'},
              node_color=[.3,.3,.3,.7,.7,.7,.7],vmin=0.0,vmax=1.0,pos=pos,ax=ax[1])
    """


def nxFactorGraph(model, all_vars=False):
    """Get a networkx Graph object of the factor graph of the model

    Example:
    >>> G=gm.nxFactorGraph(model); nx.draw(G);
    """
    import networkx as nx
    G = nx.Graph()
    vNodes = [v.label for v in model.X if (all_vars or v.states > 1)]   # all or just non-trivial vars
    fNodes = ['f{:d}'.format(i) for i in range(len(model.factors))]    # strings "fi" for factors
    G.add_nodes_from( vNodes, nodetype='var')
    G.add_nodes_from( fNodes, nodetype='factor')
    for i,f in enumerate(model.factors):
      for v in f.vars:
          G.add_edge(v.label,'f{:d}'.format(i))
    return G



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
    kwargs = copy.copy(kwargs)    # make a copy of the arguments dict for mutation
    G = nx.Graph()
    G.add_nodes_from( [v.label for v in model.X if v.states > 1] )  # only non-trivial vars
    for f in model.factors:
      for v1 in f.vars:
        for v2 in f.vars:
          if (v1 != v2): G.add_edge(v1.label,v2.label)
    kwargs['var_labels'] = kwargs.get('var_labels',{n:n for n in G.nodes()})
    kwargs['labels'] = kwargs.get('labels', kwargs.get('var_labels',{}) )
    kwargs.pop('var_labels',None)   # remove artificial "var_labels" entry)
    kwargs['edgecolors'] = kwargs.get('edgecolors','k')
    nx.draw(G,**kwargs)
    return G


def drawFactorGraph2(model, voptions=None,foptions=None,eoptions=None, **kwargs):
    """Draw a factorgraph using networkx function calls

    Args:
      voptions / foptions / eoptions [dict] : options for vars / factors / edges in draw()
      ``**kwargs``: remaining keyword arguments passed to networkx.draw()

    Example:

    >>> gm.drawFactorGraph(model, voptions={labels:{0:'0', ... }} ) # labels for var nodes 
    """
    import networkx as nx
    import copy
    kwargs = copy.copy(kwargs)    # make a copy of the arguments dict for mutation
    G = nx.Graph()
    vNodes = [v.label for v in model.X if v.states > 1]   # list only non-trivial variables
    fNodes = [-i-1 for i in range(len(model.factors))]    # use negative IDs for factors: TODO switch to "fi"?
    G.add_nodes_from( vNodes + fNodes )
    for i,f in enumerate(model.factors):
      for v in f.vars:
        G.add_edge(v.label,-i-1)
    #
    # Get default appearances for vars; add default labels; override / augment with "voptions"
    vopt = copy.copy(nxDefaults.var); vopt['labels']=[v.label for v in model.X if v.states>1]
    for arg in voptions: vopt[arg] = voptions[arg]
    # Get default appearances for fact; add default labels; override / augment with "foptions"
    fopt = copy.copy(nxDefaults.factor); fopt['labels']={n:n for n in vNodes}
    for arg in foptions: fopt[arg] = foptions[arg]
    # Get default appearances for edges; override / augment with "eoptions"
    eopt = copy.copy(nxDefaults.edge); 
    for arg in eoptions: eopt[arg] = eoptions[arg]

    if not 'pos' in kwargs: kwargs['pos'] = nx.spring_layout(G) # so we can use same positions multiple times...

    nx.draw_networkx_nodes(G, nodelist=vNodes,**vopt,**kwargs)
    nx.draw_networkx_nodes(G, nodelist=fNodes,**fopt,**kwargs)
    nx.draw_networkx_edges(G, **eopt,**kwargs)
    return G




def drawFactorGraph(model,var_color='w',factor_color=[(.2,.2,.8)],**kwargs):
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
    import copy
    kwargs = copy.copy(kwargs)    # make a copy of the arguments dict for mutation
    G = nx.Graph()
    vNodes = [v.label for v in model.X if v.states > 1]   # list only non-trivial variables
    fNodes = [-i-1 for i in range(len(model.factors))]    # use negative IDs for factors: TODO switch to "fi"?
    G.add_nodes_from( vNodes )
    G.add_nodes_from( fNodes )
    for i,f in enumerate(model.factors):
      for v1 in f.vars:
        G.add_edge(v1.label,-i-1)

    if not 'pos' in kwargs: kwargs['pos'] = nx.spring_layout(G) # so we can use same positions multiple times...
    var_labels  = kwargs.get('var_labels',kwargs.get('labels', {n:n for n in vNodes}))
    kwargs.pop('var_labels',None)   # remove artificial "var_labels" entry)
    kwargs.pop('labels',None)       #   also labels if it exists
    factor_labels = kwargs.get('factor_labels',{}); kwargs.pop('factor_labels',None);
    edge_color = kwargs.get('edge_color','k'); kwargs.pop('edge_color',None);  # colors of graph edges
    edgecolors = kwargs.get('edgecolors','k'); kwargs.pop('edgecolors',None);  # outlines of nodes
    nx.draw_networkx(G, nodelist=vNodes,node_color=var_color,edgecolors=edgecolors,labels=var_labels,**kwargs)
    # TODO: no longer accepts labels of subset of nodes?
    nx.draw_networkx_nodes(G, nodelist=fNodes,edgecolors=edgecolors,node_color=factor_color,node_shape='s',**kwargs)
    nx.draw_networkx_edges(G,edge_color=edge_color,**kwargs)
    return G



def drawBayesNet(model,**kwargs):
    """Draw a Bayesian Network (directed acyclic graph) using networkx function calls

    Args:
      ``**kwargs``: remaining keyword arguments passed to networkx.draw()

    Example:

    >>> gm.drawBayesNet(model, labels={0:'0', ... } )    # keyword args passed to networkx.draw()
    """
    import networkx as nx
    import copy
    kwargs = copy.copy(kwargs)    # make a copy of the arguments dict for mutation
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
    kwargs.pop('var_labels',None)   # remove artificial "var_labels" entry)
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
    import copy
    kwargs = copy.copy(kwargs)    # make a copy of the arguments dict for mutation
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
    labels = kwargs.get('labels',{}); kwargs.pop('labels',None);
    arrowsize = kwargs.get('arrowsize',None); kwargs.pop('arrowsize',None);
    for d in decisions: nx.draw_networkx_nodes(G, nodelist=[d], node_color=[(.7,.7,.9)], node_shape='s', **kwargs)
    for c in chance: nx.draw_networkx_nodes(G, nodelist=[c], node_color=[(1.,1.,1.)], node_shape='o', **kwargs)
    for u in util: nx.draw_networkx_nodes(G, nodelist=[u], node_color=[(.7,.9,.7)], node_shape='d', **kwargs)
    nx.draw_networkx_edges(G, edgelist=cpd_edges+util_edges, **kwargs)
    if info_edges:
      tmp = nx.draw_networkx_edges(G, edgelist=info_edges,arrowsize=arrowsize, **kwargs)
      for line in tmp: line.set_linestyle('dashed')
    nx.draw_networkx_labels(G, labels=labels, **kwargs)



#def drawJGraph(



