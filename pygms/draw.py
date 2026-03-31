from .factor import *
from .misc import *
from .graphmodel import *
import networkx as nx

class nxDefaults:
  """Set networkx drawing default appearances for variable nodes, factor nodes, and edges"""
  var = {'edgecolors':'black', 'node_color':'w', 'node_shape':'o' }
  factor = { 'edgecolors':'black', 'node_color':[(.2,.2,.8)], 'node_shape':'s' }
  edge = {'edge_color':'black' }
  util = {'node_color': [(.7,.9,.7)], 'node_shape':'d' }



def nxMarkovGraph(model, all_vars=False):
    """Get networkx Graph object of the Markov graph of the model

    Example:
    >>> G = gm.draw.nxMarkovGraph(model)
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
    >>> G=gm.draw.nxFactorGraph(model); nx.draw(G);
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


def nxBayesNet(model, all_vars=False):
    """Get networkx DiGraph representation of a Bayes net graphical model

    Example:
    >>> G = gm.draw.nxBayesNet(model)
    >>> nx.draw(G)
    """
    topo_order = bnOrder(model)                              # TODO: allow user-provided order?
    if topo_order is None: raise ValueError('Topo order not found; model is not a Bayes Net?')
    pri = np.zeros((max(topo_order)+1,))-1
    pri[topo_order] = np.arange(len(topo_order))
    G = nx.DiGraph()
    G.add_nodes_from( [v.label for v in model.X if (all_vars or v.states > 1)] ) # all or non-trivial vars
    for f in model.factors:
      v2label = topo_order[ int(max(pri[v.label] for v in f.vars)) ]
      for v1 in f.vars:
        if (v1.label != v2label): G.add_edge(v1.label,v2label)
    return G




def drawMarkovGraph(model,**kwargs):
    """Draw a Markov random field using networkx function calls

    Args:
      ``**kwargs``: remaining keyword arguments passed to networkx.draw()

    Example:

    >>> gm.drawMarkovGraph(model, labels={0:'0', ... } )    # keyword args passed to networkx.draw()
    """
    # TODO: fix defaults; specify shape, size etc. consistent with FG version
    import copy
    kwargs = copy.copy(kwargs)    # make a copy of the arguments dict for mutation
    G = nxMarkovGraph(model)

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
    G = nxFactorGraph(model)
    vNodes = [n for n,d in G.nodes(data=True) if d['nodetype']=='var']
    fNodes = [n for n,d in G.nodes(data=True) if d['nodetype']=='factor']


    # Get default appearances for vars; add default labels; override / augment with "voptions"
    vopt = copy.copy(nxDefaults.var); vopt['labels']=[v.label for v in model.X if v.states>1]
    for arg in voptions: vopt[arg] = voptions[arg]
    # Get default appearances for fact; add default labels; override / augment with "foptions"
    fopt = copy.copy(nxDefaults.factor); fopt['labels']={n:n for n in vNodes}
    for arg in foptions: fopt[arg] = foptions[arg]
    # Get default appearances for edges; override / augment with "eoptions"
    eopt = copy.copy(nxDefaults.edge); 
    for arg in eoptions: eopt[arg] = eoptions[arg]

    if not 'pos' in kwargs: kwargs['pos'] = nx.spring_layout(G) # need same positions multiple times...

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
    G = nxFactorGraph(model)
    vNodes = [n for n,d in G.nodes(data=True) if d['nodetype']=='var']
    fNodes = [n for n,d in G.nodes(data=True) if d['nodetype']=='factor']

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


#
# TODO: "generic" draw function that discerns MRF,BN,FG,ID from nxNNN() function output?
#


def drawBayesNet(model,**kwargs):
    """Draw a Bayesian Network (directed acyclic graph) using networkx function calls

    Args:
      ``**kwargs``: remaining keyword arguments passed to networkx.draw()

    Example:

    >>> gm.drawBayesNet(model, labels={0:'0', ... } )    # keyword args passed to networkx.draw()
    """
    import copy
    kwargs = copy.copy(kwargs)    # make a copy of the arguments dict for mutation
    G = nxBayesNet(model)

    kwargs['var_labels'] = kwargs.get('var_labels',{n:n for n in [v.label for v in model.X if v.states>1]})
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
    pri = np.zeros((max(topo_order)+1,))-1
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




#### TODO: fix & test
#def drawSearchTree(root):
#    """G, pos = drawSearchTree(root) :  return nxGraph of a current search tree, and position information."""
#    nodes = [[(-root.value,root)],[]]
#    while True:
#        for n in nodes[-2]:
#            nodes[-1] += n[1].children
#        if len(nodes[-1])==0: break
#        nodes.append([])
#    pos = {}
#    for d in range(len(nodes)-1,-1,-1):
#        p = 0.
#        for vn,n in nodes[d]:
#            if len(n.children)>0:
#                p = np.mean([pos[c][0] for vc,c in n.children])
#                pos[n] = (p, -d*1.)
#                p = np.max([pos[c] for vc,c in n.children]) + 1.
#            else:
#                pos[n] = (p,-d*1.)
#                p += 1
#    G = nx.Graph(); G.add_nodes_from([n for ns in nodes for n in ns]);
#    G.add_edges_from([(p,n) for ns in nodes[:-1] for p in ns for n in p.children])
#    return G,pos

# def nxTreeLayout(G):
#     """pos = nxTreeLayout(G) :  return positions for root-down layout of a tree."""
#     depth = {}; by_depth = {}
#     for n in G.nodes: 
#         dn = n.count(',')+n.count('.')  # number of variables & number of values    
#         depth[n] = dn
#         by_depth[dn] = [n] if dn not in by_depth else by_depth[dn]+[n]
# 
#     pos = {}
#     leaves = sorted([n for n in G.nodes if len(G.out_edges(n))==0])
#     max_depth = max(by_depth.keys());
#     x = -1.
#     for i,l in enumerate(leaves): 
#         if i>0:
#             x += 1. + np.log2(depth[l] - depth[nx.lowest_common_ancestor(G,l,leaves[i-1])])
#         else: x += 1. + 0*np.log2(max_depth - depth[l]+1)
#         pos[l] = (x,-depth[l]);
#         
#     for d in range(max_depth,-1,-1):
#         if d not in by_depth: continue
#         for n in by_depth[d]: 
#             if n not in pos: pos[n] = (np.mean([pos[c[1]][0] for c in G.out_edges(n)]), -depth[n])
#     return pos
# 
## TODO: nxTuneTreeLayout? Make tree layout look nicer?

def nxTreeLayout(G, pos=None):
    """Lay out a directed tree
      G (networkx.DiGraph) : directed graph; use G.reverse() if edges oriented toward root 
      pos (dict[node:(x,y)], default None) : initial node locations to adjust
    """
    nodes = sorted(G.nodes)
    roots = [n for n in G.nodes if G.in_degree[n]==0] 
    if pos is None:
      dists = tuple(d2t([nx.shortest_path_length(G, source=r)],nodes)[0] for r in roots)
      dists = dists[0] if len(dists)==1 else np.fmin( *dists )
      dists = t2d([dists],nodes)[0]
      pos = {n:(0,dists[n]) for n in nodes}   # no info given? just place them at depth d
    else: 
      pos = pos.copy()
   
    def _revise(G, root, w=1., x=0.5, pos=None, par=None):
        if type(root) is list: children = root
        else:
            pos[root] = (np.round(x,4), pos[root][1])
            children = list(G.successors(root))
        if len(children):
            children = [n[1] for n in sorted([(pos[c],c) for c in children])]  # sort by current x-vals
            dx = w/(len(children))
            x_ = x - w/2 - dx/2
            for ch in children:
                x_ += dx
                pos = _revise(G,ch, w=dx, x=x_, pos=pos, par=root)
        return pos
    pos = _revise(G,roots,pos=pos)

    # Now revise x-positions more uniformly
    xvals = np.unique([pos[p][0] for p in pos])
    xmap = {x:i*1./len(xvals) for i,x in enumerate(xvals)}
    for p in pos: pos[p]=(xmap[pos[p][0]],pos[p][1])
    return pos




def nx2tikz(G,pos, precision=1, use_names=True):
    latex = "\\begin{tikzpicture} \n"
    latex+= "  \\tikzset{var/.style = {shape=circle,draw,minimum size=1.2em}} \n"
    latex+= "  \\tikzset{edge/.style = {->,> = latex,line width=0.5mm}} \n"
    nodes = list(G.nodes)
    id = lambda n: n if use_names else nodes.index(n)
    locat = np.array([pos[key] for key in nodes]).reshape(-1,2).round(precision)
    latex += "  \\foreach \\name/\\pos in {"+",".join(["{"+"{}/({},{})".format(id(nodes[i]),locat[i,0],locat[i,1])+"}" for i in range(len(locat))])+"}\n"
    latex += "    \\node[var] (\\name) at \\pos {$\\name$}; \n"
    latex += "  \\foreach \\pa/\\ch in {"+",".join(["{"+"{}/{}".format(id(e[0]),id(e[1]))+"}" for e in G.edges])+"}\n"
    latex += "    \\draw[edge] (\\pa) to (\\ch); \n"
    latex += "\\end{tikzpicture} \n"
    return latex

